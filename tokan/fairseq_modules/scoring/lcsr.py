import math

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.scoring import BaseScorer, register_scorer


DENOMINATOR_CHOICES = ChoiceEnum(["ref", "pred", "min", "max", "sqrt"])


def longest_common_sequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[::-1]


def deduplicate(tokens):
    dedup_tokens = []
    for i, t in enumerate(tokens):
        if i == 0 or t != tokens[i - 1]:
            dedup_tokens.append(t)
    return dedup_tokens


@dataclass
class LcsrConfig(FairseqDataclass):
    deduplicate: bool = field(default=False, metadata={"help": "Whether to deduplicate repeated tokens"})
    denominator: DENOMINATOR_CHOICES = field(
        default="sqrt",
        metadata={"help": "The denominator for the LCSR score"},
    )


@register_scorer("lcsr", dataclass=LcsrConfig)
class LcsrScorer(BaseScorer):
    def __init__(self, cfg):
        self.cfg = cfg
        self.ref = []
        self.pred = []

    def add(self, ref, pred):
        if self.cfg.deduplicate:
            ref = deduplicate(ref)
            pred = deduplicate(pred)
        self.ref.append(ref)
        self.pred.append(pred)

    def add_string(self, ref, pred):
        self.add(ref.split(), pred.split())

    def _score(self, ref, pred) -> float:
        if self.cfg.denominator == "ref":
            denominator = len(ref)
        elif self.cfg.denominator == "pred":
            denominator = len(pred)
        elif self.cfg.denominator == "min":
            denominator = min(len(ref), len(pred))
        elif self.cfg.denominator == "max":
            denominator = max(len(ref), len(pred))
        elif self.cfg.denominator == "sqrt":
            denominator = math.sqrt(len(ref) * len(pred))
        else:
            raise ValueError(f"Invalid denominator: {self.cfg.denominator}")

        return len(longest_common_sequence(ref, pred)) / denominator

    def score(self):
        return sum(self._score(ref, pred) for ref, pred in zip(self.ref, self.pred)) / len(self.ref)

    def result_string(self) -> str:
        return f"LCSR: {self.score():.2%}"
