# Modified from Matcha-TTS for TokAN project

import random
from typing import Any, Dict, Optional

import numpy as np
from kaldiio import load_mat

import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from tokan.utils import get_pylogger, parse_embed_path
from tokan.matcha.text import text_to_sequence
from tokan.matcha.utils.audio import mel_spectrogram
from tokan.matcha.utils.model import fix_len_compatibility, normalize

log = get_pylogger(__name__)


def parse_metadata(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        metadata = [line.strip().split(split_char) for line in f]
    return metadata


class TokenMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_metadata_path,
        valid_metadata_path,
        batch_size,
        num_workers,
        pin_memory,
        use_spkemb,
        use_text,
        token_rate,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        deduplicate,
        upsample_rate,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TokenMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_metadata_path,
            self.hparams.use_spkemb,
            self.hparams.use_text,
            self.hparams.token_rate,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.deduplicate,
            self.hparams.upsample_rate,
        )
        self.validset = TokenMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_metadata_path,
            self.hparams.use_spkemb,
            self.hparams.use_text,
            self.hparams.token_rate,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.deduplicate,
            self.hparams.upsample_rate,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TokenMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path,
        use_spkemb,
        use_text=False,
        token_rate=50,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        deduplicate=False,
        upsample_rate=None,
    ):
        self.metadata = parse_metadata(metadata_path)
        self.use_spkemb = use_spkemb
        self.use_text = use_text

        self.token_rate = token_rate

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.deduplicate = deduplicate
        self.upsample_rate = upsample_rate

        real_upsample_rate = sample_rate / hop_length / token_rate
        assert np.isclose(
            self.upsample_rate, real_upsample_rate, atol=1e-3
        ), f"Inaccurate upsampling rate {self.upsample_rate} vs. {upsample_rate}"
        print(f"(Token -> Frame) upsampling rate: {self.upsample_rate}")

        self.cleaners = ["english_cleaners2"]
        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}

        random.seed(seed)
        random.shuffle(self.metadata)

    def get_datapoint(self, filepath_and_text):
        utt_id, filepath, text, spkemb_path, token_str = filepath_and_text[:5]

        mel, audio = self.get_mel_audio(filepath)  # (T', C): floor(T'/k) = T/k
        token = self.get_token(token_str)  # (T/k)
        token_length = token.shape[0]

        mel = self.align_mel_to_token(mel, token_length)  # (T, C)

        # Deduplicating token-level durations
        # sum(duration) = T_y / upsample_rate
        if self.deduplicate:
            token, duration = self.deduplicate_tokens(token)  # (N)
        else:
            duration = torch.ones_like(token)

        if self.use_text:
            text, cleaned_text = self.get_text(text)
        else:
            text, cleaned_text = None, None

        if self.use_spkemb:
            spk = self.get_spkemb(spkemb_path)  # (D,)
        else:
            spk = None

        return {
            "x": token,
            "duration": duration,
            "y": mel,
            "spk": spk,
            "filepath": filepath,
            "x_text": text,
            "utt_id": utt_id,
        }

    def get_mel_audio(self, filepath):
        audio, sr = ta.load(filepath)
        if sr != self.sample_rate:
            audio = ta.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(audio)
            sr = self.sample_rate
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel, audio

    def deduplicate_tokens(self, token):
        new_token = []
        duration = []
        if len(token) == 0:
            return torch.IntTensor([]), torch.IntTensor([])
        current_token = token[0]
        count = 1

        for t in token[1:]:
            if t == current_token:
                count += 1
            else:
                new_token.append(current_token)
                duration.append(count)
                current_token = t
                count = 1
        # Append the last token and its count
        new_token.append(current_token)
        duration.append(count)
        # Convert lists to numpy arrays
        new_token = torch.IntTensor(new_token)
        duration = torch.IntTensor(duration)

        return new_token, duration

    def get_text(self, text):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        text_norm = torch.IntTensor(text_norm)

        return text_norm, cleaned_text

    def get_spkemb(self, spkemb_path):
        is_ark = parse_embed_path(spkemb_path)
        if is_ark:
            spkemb = load_mat(spkemb_path)
        else:
            spkemb = np.load(spkemb_path)
        spkemb = torch.from_numpy(spkemb).float().squeeze()

        return spkemb

    def get_token(self, token_str):
        token = token_str.split(" ")
        token = [int(t) for t in token]
        token = torch.IntTensor(token)
        return token

    def align_mel_to_token(self, mel, token_length):
        mel_length = mel.shape[1]
        target_mel_length = int(token_length * self.upsample_rate)
        diff_length = mel_length - target_mel_length
        assert diff_length >= 0 and diff_length <= self.upsample_rate
        mel = mel[:, :target_mel_length]
        return mel

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.metadata[index])
        return datapoint

    def __len__(self):
        return len(self.metadata)


class TextMelBatchCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        utt_ids = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            utt_ids.append(item["utt_id"])
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            durations[i, : item["duration"].shape[-1]] = item["duration"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)

        if len(spks) > 0:
            spks = torch.stack(spks, dim=0)
        else:
            spks = None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "durations": durations,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "utt_ids": utt_ids,
        }
