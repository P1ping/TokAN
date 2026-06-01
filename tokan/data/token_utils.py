import numpy as np
from numba import jit


def deduplicate_tokens(tokens):
    """Deduplicate the tokens by removing consecutive duplicate tokens"""
    dedup_tokens = []
    durations = []
    current_token = None
    current_count = 0

    for t in tokens:
        if t == current_token:
            current_count += 1
        else:
            if current_token is not None:
                dedup_tokens.append(current_token)
                durations.append(current_count)
            current_token = t
            current_count = 1

    # Append the last token and its count
    if current_token is not None:
        dedup_tokens.append(current_token)
        durations.append(current_count)

    return dedup_tokens, durations


@jit(nopython=True, cache=True)
def lcs_trace_masks(seq1, seq2):
    """
    Computes the LCS and returns binary masks for both sequences indicating
    which elements are part of the optimal common subsequence.

    Returns:
        mask1 (np.array): Binary array of shape (len(seq1),)
        mask2 (np.array): Binary array of shape (len(seq2),)
    """
    m = len(seq1)
    n = len(seq2)

    # 1. Standard DP Table Construction
    # We must compute the full table to know the path
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                v1 = dp[i - 1, j]
                v2 = dp[i, j - 1]
                dp[i, j] = v1 if v1 > v2 else v2

    # 2. Backtracking to generate masks (Training Targets)
    # Initialize masks with zeros (0 = not in LCS, 1 = in LCS)
    mask1 = np.zeros(m, dtype=np.int8)
    mask2 = np.zeros(n, dtype=np.int8)

    i, j = m, n
    while i > 0 and j > 0:
        # If characters match, they are part of the LCS
        if seq1[i - 1] == seq2[j - 1]:
            # Mark the positions in the original sequences
            mask1[i - 1] = 1
            mask2[j - 1] = 1

            # Move diagonally up-left
            i -= 1
            j -= 1

        # If not a match, move in the direction of the larger DP value
        elif dp[i - 1, j] > dp[i, j - 1]:
            i -= 1  # Move up
        else:
            j -= 1  # Move left

    return mask1, mask2


@jit(nopython=True, cache=True)
def smear_lcs_weights(source_seq, lcs_mask, mode="center"):
    """
    Spreads the weight of an LCS match to identical consecutive neighbors.

    Args:
        source_seq: The input token sequence (int array)
        lcs_mask: Boolean array from your current lcs_trace_masks function
        mode: "average" (distribute weight uniformly across cluster),
              "center" (assign weight 1.0 to the center match_count positions)

    Returns:
        float32 array of weights between 0.0 and 1.0

    Example for mode="average":
        source=[1,2,2,2,9], lcs_mask from LCS with [0,2,2,8,8,4]
        → weights=[0, 0.67, 0.67, 0.67, 0]

    Example for mode="center":
        Same input → weights=[0, 1, 1, 0, 0]
        The match_count=2 positions are centered in the cluster of 3,
        with ties broken toward the left.
    """
    n = len(source_seq)
    weights = np.zeros(n, dtype=np.float32)

    i = 0
    while i < n:
        current_token = source_seq[i]

        # Find the end of this cluster of identical tokens
        j = i
        match_count = 0
        cluster_len = 0

        while j < n and source_seq[j] == current_token:
            if lcs_mask[j]:
                match_count += 1
            cluster_len += 1
            j += 1

        # If any token in this cluster was part of the LCS
        if match_count > 0:
            if mode == "center":
                # Center mode: assign weight 1.0 to the middle match_count positions
                start = (cluster_len - match_count) // 2
                for k in range(i + start, i + start + match_count):
                    weights[k] = 1.0
            elif mode == "average":
                # Average mode: distribute weight uniformly across the cluster
                weight = match_count / cluster_len
                for k in range(i, j):
                    weights[k] = weight

        # Move to next cluster
        i = j

    return weights
