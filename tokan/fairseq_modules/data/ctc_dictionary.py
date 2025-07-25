# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

from fairseq.data.dictionary import Dictionary


class CTCDictionary(Dictionary):
    def __init__(
        self,
        *,  # begin keyword-only arguments
        blank="<blank>",
        pad="<pad>",
        unk="<unk>",
        extra_special_symbols=None,
        add_special_symbols=True,
    ):
        self.blank_word, self.unk_word, self.pad_word = blank, unk, pad
        self.symbols = []
        self.count = []
        self.indices = {}
        if add_special_symbols:
            self.blank_index = self.add_symbol(blank)
            self.pad_index = self.add_symbol(pad)
            self.unk_index = self.add_symbol(unk)
            if extra_special_symbols:
                for s in extra_special_symbols:
                    self.add_symbol(s)
            self.nspecial = len(self.symbols)

    def blank(self):
        """Helper to get index of blank symbol"""
        return self.blank_index

    def bos(self):
        raise NotImplementedError("CTC does not support beginning-of-sentence symbol")

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        raise NotImplementedError("CTC does not support end-of-sentence symbol")

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index
