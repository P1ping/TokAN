# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
# Based on fairseq's speech_to_text_dataset.py
#
# Licensed under the MIT License - see LICENSE file for details

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

import torch

from fairseq.data import Dictionary
from fairseq.data import data_utils

from .speech_token_to_token_dataset import (
    SpeechTokenToTokenDataset,
    SpeechTokenToTokenDatasetCreator,
    _is_int_or_np_int,
    SpeechTokenToTokenDatasetItem,
)


logger = logging.getLogger(__name__)


class SpeechTokenDenoisingDataset(SpeechTokenToTokenDataset):
    """A dataset for speech token denoising. The noising functions are modifed from fairseq's DenoisingDataset."""

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        audio_paths: List[str] = None,
        src_texts: Optional[List[str]] = None,
        aux_texts: Optional[List[str]] = None,
        src_embeds: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        src_dict: Optional[Dictionary] = None,
        aux_dict: Optional[Dictionary] = None,
        left_pad_source: bool = False,
        left_pad_target: bool = False,
        append_eos=True,
        load_waveform=False,
        normalize_waveform=False,
        load_src_embed=False,
        load_aux_text=False,
        poisson_lambda=3.0,
        mask_ratio=0.3,
        random_ratio=0.1,
        insert_ratio=0.1,
        replace_length=1,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.audio_paths = audio_paths
        self.n_samples = len(audio_paths)
        self.mask_idx = src_dict.index("<mask>")
        assert self.mask_idx != src_dict.unk(), "mask token not in dictionary"
        self.seed = 373

        self.replace_length = replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        self.mask_span_distribution = self.build_mask_span_distribution(poisson_lambda)
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.insert_ratio = insert_ratio

        assert ids is not None and len(ids) == self.n_samples
        assert src_texts is not None and len(src_texts) == self.n_samples

        self.src_texts = src_texts
        self.aux_texts = aux_texts
        self.src_embeds = src_embeds
        self.src_dict = src_dict
        self.tgt_dict = src_dict  # for denoising, src and tgt are the same
        self.aux_dict = aux_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target

        self.ids = ids
        self.shuffle = is_train_split

        self.src_lens = self.get_src_lens_and_check_oov()
        self.aux_lens = self.get_aux_lens_and_check_oov() if load_aux_text else None

        self.append_eos = append_eos
        self.load_waveform = load_waveform
        self.normalize_waveform = normalize_waveform

        self.load_src_embed = load_src_embed
        self.load_aux_text = load_aux_text

        self.epoch = 0

        logger.info(self.__repr__())

    def build_mask_span_distribution(self, poisson_lambda):
        _lambda = poisson_lambda

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        mask_span_distribution = torch.distributions.Categorical(ps)
        return mask_span_distribution

    def word_starts(self, source):
        is_word_start = torch.ones(source.size())
        is_word_start[-1] = 0
        return is_word_start

    def add_masking_noise(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return self.add_insertion_noise(source, num_inserts / source.size(0))

        # part 2
        assert (lengths > 0).all()
        assert is_word_start[-1] == 0

        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(1, len(self.src_dict), size=(mask_random.sum(),))

        # part 3
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = self.mask_idx
                source[indices[mask_random]] = torch.randint(1, len(self.src_dict), size=(mask_random.sum(),))

        # part 4
        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.src_dict), size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def get_src_tgt_tokens(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.src_texts[index]
        else:
            text = " ".join([self.src_texts[i] for i in index])
        tokens = self.src_dict.encode_line(text, add_if_not_exist=False, append_eos=False).long()

        source, target = tokens, tokens.clone()

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            if self.mask_ratio > 0:
                source = self.add_masking_noise(source, self.mask_ratio)
            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.src_dict)).all()

        return source, target

    def __getitem__(self, index: int) -> SpeechTokenToTokenDatasetItem:
        utt_id = self.ids[index]

        src_wav = self._get_source_audio(index) if self.load_waveform else None
        src_embed = self.get_src_embed(index) if self.load_src_embed else None

        src_tokens, tgt_tokens = self.get_src_tgt_tokens(index)
        aux_text = self.get_aux_texts(index) if self.load_aux_text else None

        return SpeechTokenToTokenDatasetItem(
            index=index,
            utt_id=utt_id,
            src_wav=src_wav,
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            src_embed=src_embed,
            aux_text=aux_text,
        )

    def __len__(self):
        return self.n_samples

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def num_tokens(self, index):
        return self.src_lens[index]

    def size(self, index):
        src_length = self.src_lens[index]
        return src_length, src_length

    @property
    def sizes(self):
        return np.array(self.src_lens)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.src_lens])
        return np.lexsort(order)


class SpeechTokenDenoisingDatasetCreator(SpeechTokenToTokenDatasetCreator):
    # mandatory columns
    KEY_ID, KEY_AUDIO = "id", "src_audio"
    KEY_SRC_TEXT, KEY_TGT_TEXT = "src_tokens", "tgt_tokens"
    # optional columns
    KEY_SRC_EMB, DEFAULT_SRC_EMB = "src_embed", None
    KEY_AUX_TEXT, DEFAULT_AUX_TEXT = "aux_text", None

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        src_dict,
        aux_dict,
        left_pad_source,
        left_pad_target,
        load_waveform,
        load_src_embed,
        load_aux_text,
        poisson_lambda,
        mask_ratio,
        random_ratio,
        insert_ratio,
        replace_length,
    ) -> SpeechTokenDenoisingDataset:
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [Path(s[cls.KEY_AUDIO]).as_posix() for s in samples]
        src_texts = [s[cls.KEY_SRC_TEXT] for s in samples]
        aux_texts = [s.get(cls.KEY_AUX_TEXT, cls.DEFAULT_AUX_TEXT) for s in samples]
        src_embeds = [s.get(cls.KEY_SRC_EMB, cls.DEFAULT_SRC_EMB) for s in samples]

        ds = SpeechTokenDenoisingDataset(
            split=split_name,
            is_train_split=is_train_split,
            audio_paths=audio_paths,
            src_texts=src_texts,
            aux_texts=aux_texts,
            src_embeds=src_embeds,
            ids=ids,
            src_dict=src_dict,
            aux_dict=aux_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            load_waveform=load_waveform,
            load_src_embed=load_src_embed,
            load_aux_text=load_aux_text,
            poisson_lambda=poisson_lambda,
            mask_ratio=mask_ratio,
            random_ratio=random_ratio,
            insert_ratio=insert_ratio,
            replace_length=replace_length,
        )
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        split: str,
        is_train_split: bool,
        src_dict: Dictionary,
        aux_dict: Dictionary,
        left_pad_source: bool,
        left_pad_target: bool,
        load_waveform: bool,
        load_src_embed: bool,
        load_aux_text: bool,
        poisson_lambda: float,
        mask_ratio: float,
        random_ratio: float,
        insert_ratio: float,
        replace_length: int,
    ) -> SpeechTokenDenoisingDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            src_dict,
            aux_dict,
            left_pad_source,
            left_pad_target,
            load_waveform,
            load_src_embed,
            load_aux_text,
            poisson_lambda,
            mask_ratio,
            random_ratio,
            insert_ratio,
            replace_length,
        )
