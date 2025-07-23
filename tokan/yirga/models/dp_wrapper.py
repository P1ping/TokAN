from typing import Dict

import torch
import torch.nn.functional as F

from einops import reduce

from tokan.utils import get_pylogger
from tokan.matcha.utils.model import (
    generate_path,
    sequence_mask,
)

from tokan.matcha.utils.utils import plot_tensor
from tokan.yirga.models.baselightningmodule import BaseLightningClass
from tokan.yirga.models.components.variance_predictor import get_duration_predictor
from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel

log = get_pylogger(__name__)


class DurationPredictorWrapper(BaseLightningClass):  # ☕
    def __init__(
        self,
        yirga_checkpoint,
        duration_predictor_params,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()

        ttm_model = YirgaTokenToMel.load_from_checkpoint(yirga_checkpoint, map_location="cpu")
        for param in ttm_model.parameters():
            param.requires_grad = False
        ttm_model.eval()

        duration_predictor_params.n_channels = ttm_model.encoder.n_channels
        duration_predictor = get_duration_predictor(duration_predictor_params)

        self.ttm_model = self.merge_model_with_dp(ttm_model, duration_predictor, duration_predictor_params)
        self.ttm_model.save_hyperparameters(logger=False)
        self.save_hyperparameters(self.ttm_model.hparams, logger=False)
        self.ttm_model.duration_predictor.train()

    @torch.inference_mode()
    def synthesize(self, x, x_lengths, spks, total_duration):
        x, x_mask = self.ttm_model.encoder(x, x_lengths, spks)
        spk_embed = self.ttm_model.spk_embedder(spks)  # (B, D)
        x = x + spk_embed.unsqueeze(-1) * x_mask  # (B, D, T)
        d = self.ttm_model.duration_predictor(x, x_mask, total_duration)
        return d, x_mask

    def forward(self, x, x_lengths, spks, durations, y=None, y_lengths=None, cond=None):
        x, x_mask = self.ttm_model.encoder(x, x_lengths, spks)

        spk_embed = self.ttm_model.spk_embedder(spks)  # (B, D)
        x = x + spk_embed.unsqueeze(-1) * x_mask  # (B, D, T)

        attn = self.get_hard_attn(durations, x_mask)
        d = torch.sum(attn.unsqueeze(1), -1) * x_mask

        loss = self.ttm_model.duration_predictor.compute_loss(x.detach(), x_mask, d)

        return loss

    def get_hard_attn(self, durations, x_mask):
        """
        Args:
            durations (torch.Tensor): (B, T_x)
            x_mask (torch.Tensor): (B, 1, T_x)

        Returns:
            attn (torch.Tensor): (B, T_x, T_y)
        """
        x_dedup_lengths = durations.sum(dim=1).long()
        x_dedup_mask = sequence_mask(x_dedup_lengths, x_dedup_lengths.max()).unsqueeze(1).to(x_mask)
        attn_dedup_mask = x_mask.unsqueeze(-1) * x_dedup_mask.unsqueeze(2)
        attn_dedup = generate_path(durations, attn_dedup_mask.squeeze(1))  # (B, T_x, T_y/k)
        attn = F.interpolate(attn_dedup.float(), scale_factor=self.ttm_model.upsample_rate, mode="nearest")
        return attn

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        spks = batch["spks"]
        durations = batch["durations"]

        loss = self(
            x=x,
            x_lengths=x_lengths,
            spks=spks,
            durations=durations,
            y=batch["y"],
            y_lengths=batch["y_lengths"]
        )
        return {
            "dur_loss": loss,
        }

    def training_step(self, batch: Dict, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Dict, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        pass
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    durations = one_batch["durations"][i].unsqueeze(0).to(self.device)  # (B, T_x)
                    x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)  # (B)
                    x_max_length = x_lengths.max()
                    durations = durations[:, :x_max_length]
                    x_mask = sequence_mask(x_lengths, x_max_length).unsqueeze(1).float().to(self.device)  # (B, 1, T_x)
                    attn = self.get_hard_attn(durations, x_mask)  # (B, T_x, T_y)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(attn.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                durations = one_batch["durations"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None

                total_duration = durations.sum(dim=1) * self.ttm_model.upsample_rate
                d, x_mask = self.synthesize(x, x_lengths, spks, total_duration)  # (B, 1, T_x), (B, 1, T_x)

                gen_total_duration = reduce(d, "b 1 t -> b", "sum").ceil().long()
                y_mask = (
                    sequence_mask(gen_total_duration, gen_total_duration.max()).unsqueeze(1).float().to(self.device)
                )
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = generate_path(d.squeeze(1), attn_mask.squeeze(1))  # (B, T_x, T_y/k)
                attn = attn[0, : x_lengths[0], : gen_total_duration[0]]  # (T_x, T_y) after truncation
                self.logger.experiment.add_image(
                    f"synthesized/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def merge_model_with_dp(self, ttm_model, dp_model, dp_params):
        ttm_model.duration_predictor = dp_model
        ttm_model.hparams["encoder"]["duration_predictor_params"] = dp_params
        return ttm_model

    def state_dict(self):
        return self.ttm_model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.ttm_model.load_state_dict(state_dict, strict=strict)
        return self
