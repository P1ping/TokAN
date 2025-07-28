# Most code from fairseq/tasks/translation.py

from dataclasses import dataclass, field
import logging
import os
from typing import Optional

from fairseq import utils
from fairseq.data import (
    data_utils,
)

from ..data.speech_token_to_token_dataset import (
    SpeechTokenToTokenDataset,
    SpeechTokenToTokenDatasetCreator,
)
from ..data.ctc_dictionary import CTCDictionary
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@dataclass
class SpeechTokenTranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={"help": "manifest root path"},
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    aux_text_tag: Optional[str] = field(
        default="aux",
        metadata={"help": "tag for auxiliary text"},
    )
    left_pad_source: bool = field(default=False, metadata={"help": "pad the source on the left"})
    left_pad_target: bool = field(default=False, metadata={"help": "pad the target on the left"})
    max_source_positions: int = field(default=1024, metadata={"help": "max number of tokens in the source sequence"})
    max_target_positions: int = field(default=1024, metadata={"help": "max number of tokens in the target sequence"})
    load_waveform: bool = field(default=False, metadata={"help": "whether to load additional waveform"})
    load_src_embed: bool = field(default=False, metadata={"help": "whether to load source embedding"})
    load_aux_text: bool = field(default=False, metadata={"help": "whether to load auxiliary text"})


@register_task("speech_token_to_token", dataclass=SpeechTokenTranslationConfig)
class SpeechTokenTranslation(FairseqTask):
    """
    Translate speech tokens from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: SpeechTokenTranslationConfig

    def __init__(self, cfg: SpeechTokenTranslationConfig, src_dict, tgt_dict, aux_dict=None):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.aux_dict = aux_dict

    @classmethod
    def setup_task(cls, cfg: SpeechTokenTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception("Could not infer language pair, please provide it explicitly")

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        if cfg.load_aux_text:
            aux_dict = cls.load_ctc_dictionary(os.path.join(paths[0], f"dict.{cfg.aux_text_tag}.txt"))
            logger.info(f"[{cfg.aux_text_tag}] dictionary: {len(aux_dict)} types (blank_idx={aux_dict.blank()})")
        else:
            aux_dict = None

        return cls(cfg, src_dict, tgt_dict, aux_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechTokenToTokenDatasetCreator.from_tsv(
            root=self.cfg.data,
            split=split,
            is_train_split=is_train_split,
            src_dict=self.src_dict,
            tgt_dict=self.tgt_dict,
            aux_dict=self.aux_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            load_waveform=self.cfg.load_waveform,
            load_src_embed=self.cfg.load_src_embed,
            load_aux_text=self.cfg.load_aux_text,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, audio_paths, constraints=None, **kwargs):
        return SpeechTokenToTokenDataset(
            "interactive",
            False,
            audio_paths,
            src_tokens,
            src_lengths,
        )

    def build_model(self, cfg, from_checkpoint=False):
        if cfg.load_aux_text:
            cfg.num_ctc_classes = len(self.aux_dict)
            logger.info(f"Update model config with num_classes={cfg.num_ctc_classes}")
        model = super().build_model(cfg, from_checkpoint)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def auxiliary_dictionary(self):
        """Return the auxiliary :class:`~fairseq.data.Dictionary`."""
        return self.aux_dict

    @classmethod
    def load_ctc_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = CTCDictionary.load(filename)
        return dictionary
