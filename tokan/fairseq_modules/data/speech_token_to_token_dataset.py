# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
# Based on fairseq's speech_to_text_dataset.py
#
# Licensed under the MIT License - see LICENSE file for details

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple, Union

import numpy as np
from kaldiio.matio import load_mat

import torch
import torch.nn.functional as F

from fairseq.data import Dictionary, FairseqDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.speech_to_text_dataset import _collate_frames


logger = logging.getLogger(__name__)


def _is_int_or_np_int(n) -> bool:
    return isinstance(n, int) or (isinstance(n, np.generic) and isinstance(n.item(), int))


def parse_path(path: str) -> bool:
    """Parse data path which is either a path to
    1. a NPY file
    2. an ARK file with slicing info: "[ark_path]:[offset]"
    (Adapted from fairseq.data.audio.audio_utils.parse_path)

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          is_ark (bool): whether the path is an ARK file
    """
    suffix = Path(path).suffix
    if suffix == ".npy":
        _path, is_ark = path, False
    else:
        _path, *slice_ptr = path.split(":")
        new_suffix = Path(_path).suffix
        assert len(slice_ptr) == 1, f"Invalid ARK path: {path}"
        assert new_suffix == ".ark", f"Invalid ARK path: {path}"
        is_ark = True
    if not Path(_path).is_file():
        raise FileNotFoundError(f"File not found: {_path}")
    return is_ark


class SpeechTokenToTokenDatasetItem(NamedTuple):
    index: int
    utt_id: str
    src_tokens: torch.Tensor
    tgt_tokens: torch.Tensor
    aux_text: torch.Tensor = None
    src_wav: Optional[torch.Tensor] = None
    src_embed: Optional[torch.Tensor] = None


class SpeechTokenToTokenDataset(FairseqDataset):

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        audio_paths: List[str],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        aux_texts: Optional[List[str]] = None,
        src_embeds: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        src_dict: Optional[Dictionary] = None,
        tgt_dict: Optional[Dictionary] = None,
        aux_dict: Optional[Dictionary] = None,
        left_pad_source: bool = False,
        left_pad_target: bool = False,
        append_eos=True,
        load_waveform=False,
        normalize_waveform=False,
        load_src_embed=False,
        load_aux_text=False,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.audio_paths = audio_paths
        self.n_samples = len(audio_paths)

        assert ids is not None and len(ids) == self.n_samples
        assert src_texts is not None and len(src_texts) == self.n_samples
        assert tgt_texts is not None and len(tgt_texts) == self.n_samples
        assert src_dict is not None and tgt_dict is not None

        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.aux_texts = aux_texts
        self.src_embeds = src_embeds
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.aux_dict = aux_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target

        self.ids = ids
        self.shuffle = is_train_split

        self.src_lens = self.get_src_lens_and_check_oov()
        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.aux_lens = self.get_aux_lens_and_check_oov() if load_aux_text else None

        self.append_eos = append_eos
        self.load_waveform = load_waveform
        self.normalize_waveform = normalize_waveform

        self.load_src_embed = load_src_embed
        self.load_aux_text = load_aux_text

        logger.info(self.__repr__())

    def get_src_lens_and_check_oov(self):
        src_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.src_texts[i].split(" ")
            oov_tokens = [t for t in tokenized if self.src_dict.index(t) == self.src_dict.unk_index]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            src_lens.append(len(tokenized))
        logger.info(f"'{self.split}' source has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return src_lens

    def get_tgt_lens_and_check_oov(self):
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.tgt_texts[i].split(" ")
            oov_tokens = [t for t in tokenized if self.tgt_dict.index(t) == self.tgt_dict.unk_index]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}' target has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def get_aux_lens_and_check_oov(self):
        aux_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.aux_texts[i].split(" ")
            oov_tokens = [t for t in tokenized if self.aux_dict.index(t) == self.aux_dict.unk_index]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            aux_lens.append(len(tokenized))
        logger.info(f"'{self.split}' aux has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return aux_lens

    def __repr__(self):
        return (
            self.__class__.__name__ + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"shuffle={self.shuffle}, "
        )

    def get_src_tokens(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.src_texts[index]
        else:
            text = " ".join([self.src_texts[i] for i in index])
        tokens = self.src_dict.encode_line(text, add_if_not_exist=False, append_eos=False).long()
        return tokens

    def get_tgt_tokens(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.tgt_texts[index]
        else:
            text = " ".join([self.tgt_texts[i] for i in index])
        tokens = self.tgt_dict.encode_line(text, add_if_not_exist=False, append_eos=self.append_eos).long()
        return tokens

    def get_aux_texts(self, index: Union[int, List[int]]):
        if _is_int_or_np_int(index):
            text = self.aux_texts[index]
        else:
            text = " ".join([self.aux_texts[i] for i in index])
        tokens = self.aux_dict.encode_line(text, add_if_not_exist=False, append_eos=False).long()
        return tokens

    def get_src_embed(self, index: int):
        assert _is_int_or_np_int(index)
        embed_path = self.src_embeds[index]
        is_ark = parse_path(embed_path)
        if is_ark:
            embed = load_mat(embed_path)
        else:
            embed = np.load(embed_path)
        return torch.from_numpy(embed).float()

    def _get_source_audio(self, index: Union[int, List[int]]) -> torch.Tensor:
        """
        Gives source audio for given index with any relevant transforms
        applied. For ConcatAug, source audios for given indices are
        concatenated in given order.
        Args:
            index (int or List[int]): index—or in the case of ConcatAug,
            indices—to pull the source audio for
        Returns:
            source audios concatenated for given indices with
            relevant transforms appplied
        """
        if _is_int_or_np_int(index):
            source = get_features_or_waveform(
                self.audio_paths[index],
                need_waveform=True,
                use_sample_rate=16000,
            )
        else:
            source = np.concatenate(
                [
                    get_features_or_waveform(
                        self.audio_paths[i],
                        need_waveform=True,
                        use_sample_rate=16000,
                    )
                    for i in index
                ]
            )
        source = torch.from_numpy(source).float()
        if self.normalize_waveform:
            with torch.no_grad():
                source = F.layer_norm(source, source.shape)
        return source

    def __getitem__(self, index: int) -> SpeechTokenToTokenDatasetItem:
        utt_id = self.ids[index]

        src_wav = self._get_source_audio(index) if self.load_waveform else None
        src_embed = self.get_src_embed(index) if self.load_src_embed else None

        src_tokens = self.get_src_tokens(index)
        tgt_tokens = self.get_tgt_tokens(index)
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

    def collater(self, samples: List[SpeechTokenToTokenDatasetItem], return_order: bool = False) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        src_wavs = [x.src_wav for x in samples]
        src_embeds = [x.src_embed for x in samples]
        src_tokens = [x.src_tokens for x in samples]
        tgt_tokens = [x.tgt_tokens for x in samples]
        aux_texts = [x.aux_text for x in samples]

        n_tokens = torch.tensor([x.size(0) for x in src_tokens], dtype=torch.long)
        n_tokens, order = n_tokens.sort(descending=True)
        indices = indices.index_select(0, order)

        utt_ids = [samples[i].utt_id for i in order]

        if self.load_waveform:
            src_wavs = _collate_frames(src_wavs, is_audio_input=True)
            src_wavs = src_wavs.index_select(0, order)

        if self.load_src_embed:
            src_embeds = torch.stack(src_embeds)
            src_embeds = src_embeds.index_select(0, order)

        if self.load_aux_text:
            aux_texts = fairseq_data_utils.collate_tokens(
                aux_texts,
                self.aux_dict.pad(),
                eos_idx=None,
                left_pad=self.left_pad_source,
                move_eos_to_beginning=False,
            )
            aux_texts = aux_texts.index_select(0, order)
            aux_lengths = torch.tensor([x.aux_text.size(0) for x in samples], dtype=torch.long).index_select(0, order)

        src_tokens = fairseq_data_utils.collate_tokens(
            [x.src_tokens for x in samples],
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=self.left_pad_source,
            move_eos_to_beginning=False,
        )
        src_tokens = src_tokens.index_select(0, order)
        source_lengths = torch.tensor([x.src_tokens.size(0) for x in samples], dtype=torch.long).index_select(0, order)

        tgt_tokens = fairseq_data_utils.collate_tokens(
            [x.tgt_tokens for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=self.left_pad_target,
            move_eos_to_beginning=False,
        )
        tgt_tokens = tgt_tokens.index_select(0, order)
        target_lengths = torch.tensor([x.tgt_tokens.size(0) for x in samples], dtype=torch.long).index_select(0, order)
        prev_output_tokens = fairseq_data_utils.collate_tokens(
            [x.tgt_tokens for x in samples],
            self.tgt_dict.pad(),
            eos_idx=None,
            left_pad=self.left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.tgt_tokens.size(0) for x in samples)

        net_input = {
            "src_tokens": src_tokens,
            "src_lengths": source_lengths,
            "prev_output_tokens": prev_output_tokens,
            "src_wavs": src_wavs if self.load_waveform else None,
            "condition_embeds": src_embeds if self.load_src_embed else None,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "target": tgt_tokens,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "utt_ids": utt_ids,
            "aux_texts": aux_texts if self.load_aux_text else None,
            "aux_lengths": aux_lengths if self.load_aux_text else None,
        }
        if return_order:
            out["order"] = order
        return out

    def num_tokens(self, index):
        return self.src_lens[index]

    def size(self, index):
        return self.src_lens[index], self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.src_lens)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.src_lens])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechTokenToTokenDatasetCreator(object):
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
        tgt_dict,
        aux_dict,
        left_pad_source,
        left_pad_target,
        load_waveform,
        load_src_embed,
        load_aux_text,
    ) -> SpeechTokenToTokenDataset:
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [Path(s[cls.KEY_AUDIO]).as_posix() for s in samples]
        src_texts = [s[cls.KEY_SRC_TEXT] for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        aux_texts = [s.get(cls.KEY_AUX_TEXT, cls.DEFAULT_AUX_TEXT) for s in samples]
        src_embeds = [s.get(cls.KEY_SRC_EMB, cls.DEFAULT_SRC_EMB) for s in samples]

        ds = SpeechTokenToTokenDataset(
            split=split_name,
            is_train_split=is_train_split,
            audio_paths=audio_paths,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            aux_texts=aux_texts,
            src_embeds=src_embeds,
            ids=ids,
            src_dict=src_dict,
            tgt_dict=tgt_dict,
            aux_dict=aux_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            load_waveform=load_waveform,
            load_src_embed=load_src_embed,
            load_aux_text=load_aux_text,
        )
        return ds

    @classmethod
    def _load_samples_from_tsv(cls, root: str, split: str):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        return samples

    @classmethod
    def from_tsv(
        cls,
        root: str,
        split: str,
        is_train_split: bool,
        src_dict: Dictionary,
        tgt_dict: Dictionary,
        aux_dict: Dictionary,
        left_pad_source: bool,
        left_pad_target: bool,
        load_waveform: bool,
        load_src_embed: bool,
        load_aux_text: bool,
    ) -> SpeechTokenToTokenDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            src_dict,
            tgt_dict,
            aux_dict,
            left_pad_source,
            left_pad_target,
            load_waveform,
            load_src_embed,
            load_aux_text,
        )
