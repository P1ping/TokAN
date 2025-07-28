# Most code from fairseq/tasks/translation.py

from dataclasses import dataclass, field
import logging
import os
from omegaconf import II

from fairseq import utils
from fairseq.tasks import register_task

from ..data.speech_token_denoising_dataset import SpeechTokenDenoisingDatasetCreator
from .speech_token_to_token import SpeechTokenTranslation, SpeechTokenTranslationConfig


logger = logging.getLogger(__name__)


@dataclass
class SpeechTokenDenoisingConfig(SpeechTokenTranslationConfig):
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "lambda for Poisson distribution"},
    )
    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "Propotion of tokens to mask"},
    )
    random_ratio: float = field(
        default=0.1,
        metadata={
            "help": "probability of replacing a mask token with a random token. P(random token) = `random_ratio`; P(<mask>) = `1-random_ratio`"
        },
    )
    insert_ratio: float = field(
        default=0.1,
        metadata={"help": "probability of inserting a token"},
    )
    replace_length: int = field(
        default=1,
        metadata={
            "help": "The replacement length of each span. `-1`: an originally-lengthed segment; `0`: deletion; `1`: a single token."
        },
    )
    seed: int = II("common.seed")


@register_task("speech_token_denoising", dataclass=SpeechTokenDenoisingConfig)
class SpeechTokenDenoising(SpeechTokenTranslation):
    """
    Denoising task for speech tokens.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source speech
    """

    cfg: SpeechTokenDenoisingConfig

    def __init__(self, cfg: SpeechTokenDenoisingConfig, src_dict, aux_dict=None):
        super().__init__(cfg, src_dict, src_dict, aux_dict)

    @classmethod
    def setup_task(cls, cfg: SpeechTokenDenoisingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        assert cfg.source_lang is not None, f"source_lang is None"

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)))
        mask_idx = src_dict.add_symbol("<mask>")
        logger.info("[{}] dictionary: {} types, mask_idx: {}".format(cfg.source_lang, len(src_dict), mask_idx))

        if cfg.load_aux_text:
            aux_dict = cls.load_ctc_dictionary(os.path.join(paths[0], f"dict.{cfg.aux_text_tag}.txt"))
            logger.info(f"[{cfg.aux_text_tag}] dictionary: {len(aux_dict)} types (blank_idx={aux_dict.blank()})")
        else:
            aux_dict = None

        return cls(cfg, src_dict, aux_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechTokenDenoisingDatasetCreator.from_tsv(
            root=self.cfg.data,
            split=split,
            is_train_split=is_train_split,
            src_dict=self.src_dict,
            aux_dict=self.aux_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            load_waveform=self.cfg.load_waveform,
            load_src_embed=self.cfg.load_src_embed,
            load_aux_text=self.cfg.load_aux_text,
            poisson_lambda=self.cfg.poisson_lambda,
            mask_ratio=self.cfg.mask_ratio,
            random_ratio=self.cfg.random_ratio,
            insert_ratio=self.cfg.insert_ratio,
            replace_length=self.cfg.replace_length,
        )
