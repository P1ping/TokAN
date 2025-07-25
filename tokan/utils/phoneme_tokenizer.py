# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import re

from collections import defaultdict, OrderedDict
from typing import Optional, Union, List, Pattern

from phonemizer.phonemize import _phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import logging

from tokan.utils import get_pylogger

logger = get_pylogger(__name__)


class PhonemeTokenizer:
    def __init__(self, trace_freq: bool = True, vocab_file: Optional[str] = None):
        preserve_punctuation = False
        with_stress = True
        self.separator = Separator(phone=" ", syllable="", word=" _ ")

        # Surpass the espeak logger to avoid the warning message
        espeak_logger = logging.getLogger("espeak")
        espeak_logger.setLevel(logging.ERROR)
        self.espeak_backend = EspeakBackend(
            language="en-us",
            punctuation_marks=Punctuation.default_marks(),
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=False,
            language_switch="remove-flags",
            words_mismatch="ignore",
            logger=espeak_logger,
        )

        self.trace_freq = trace_freq
        self.phn2freq = defaultdict(int)
        if trace_freq:
            logger.info("Phoneme frequency will be traced.")

        if vocab_file is not None:
            self.load_vocab(vocab_file)
            logger.info(f"Loaded {len(self.phn2freq)} phonemes from {vocab_file}")

    def phonemize(self, text: str) -> str:
        phoneme_str = _phonemize(
            self.espeak_backend,
            text,
            self.separator,
            njobs=1,
            strip=False,
            prepend_text=False,
            preserve_empty_lines=False,
        )
        phoneme_str = self.post_process(phoneme_str)
        phonemes = phoneme_str.split(" ")
        if self.trace_freq:
            self.update_freq(phonemes)
        return phonemes

    def post_process(self, text: str, pattern: Optional[Union[str, Pattern]] = None) -> str:
        if pattern is None:
            pattern = r"\s+"
        return re.sub(pattern, " ", text).strip(self.separator.word).strip()

    def load_vocab(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            for line in f:
                phn, freq = line.strip().split()
                self.phn2freq[phn] = int(freq)

    def update_freq(self, phonemes: List[str]):
        for phn in phonemes:
            self.phn2freq[phn] += 1

    @property
    def dict(self):
        dic = OrderedDict()
        for phn, freq in sorted(self.phn2freq.items(), key=lambda x: x[1], reverse=True):
            dic[phn] = freq
        return dic
