from __future__ import annotations

from typing import Sequence

import torch


def maxsim(
    corpus: torch.Tensor,
    corpus_counts: torch.Tensor | Sequence[int],
    queries: torch.Tensor,
    block_bytes: int | None = None,
) -> torch.Tensor:
    if not isinstance(corpus, torch.Tensor):
        raise TypeError("corpus must be a torch.Tensor")
    if not isinstance(queries, torch.Tensor):
        raise TypeError("queries must be a torch.Tensor")

    if corpus.dtype != torch.float32:
        raise TypeError("corpus must be float32")
    if queries.dtype != torch.float32:
        raise TypeError("queries must be float32")
    if corpus.ndim != 2:
        raise ValueError("corpus must be 2D")
    if queries.ndim != 2:
        raise ValueError("queries must be 2D")

    if corpus.shape[1] != queries.shape[1]:
        raise ValueError("corpus and queries must have the same number of columns")
    if corpus.device != queries.device:
        raise ValueError("corpus and queries must be on the same device")

    if isinstance(corpus_counts, torch.Tensor):
        if corpus_counts.dtype != torch.int32:
            raise TypeError("corpus_counts must be int32")
        if corpus_counts.ndim != 1:
            raise ValueError("corpus_counts must be 1D")
        if corpus_counts.device not in (corpus.device, torch.device("cpu")):
            raise ValueError("corpus_counts must be on the same device as corpus or CPU")
        counts = corpus_counts.tolist()
    else:
        try:
            counts = list(corpus_counts)
        except TypeError as exc:
            raise TypeError("corpus_counts must be a torch.Tensor or a sequence of ints") from exc
    if sum(counts) != corpus.shape[0]:
        raise ValueError("corpus_counts must sum to the number of rows in corpus")
    if any(count <= 0 for count in counts):
        raise ValueError("corpus_counts entries must be positive")

    num_segments = len(counts)
    num_queries = queries.shape[0]

    if block_bytes is not None:
        if block_bytes <= 0:
            raise ValueError("block_bytes must be positive when provided")
        bytes_per_score = queries.element_size()
        max_block = block_bytes // (num_queries * bytes_per_score)
        if max_block < 1:
            max_block = 1

        max_scores = torch.full(
            (num_queries, num_segments),
            -float("inf"),
            device=queries.device,
            dtype=queries.dtype,
        )

        total_rows = corpus.shape[0]
        segment_idx = 0
        seg_start = 0
        seg_end = counts[0] if num_segments else 0
        block_start = 0

        while block_start < total_rows and segment_idx < num_segments:
            block_end = min(block_start + max_block, total_rows)
            block_scores = queries @ corpus[block_start:block_end].T

            while segment_idx < num_segments and seg_end <= block_start:
                segment_idx += 1
                if segment_idx >= num_segments:
                    break
                seg_start = seg_end
                seg_end = seg_start + counts[segment_idx]
            if segment_idx >= num_segments:
                break

            while segment_idx < num_segments and seg_start < block_end:
                local_start = seg_start if seg_start > block_start else block_start
                local_end = seg_end if seg_end < block_end else block_end
                col_start = local_start - block_start
                col_end = local_end - block_start
                segment_scores = block_scores[:, col_start:col_end]
                block_max = segment_scores.max(dim=1).values
                max_scores[:, segment_idx] = torch.maximum(
                    max_scores[:, segment_idx], block_max
                )

                if seg_end <= block_end:
                    segment_idx += 1
                    if segment_idx >= num_segments:
                        break
                    seg_start = seg_end
                    seg_end = seg_start + counts[segment_idx]
                else:
                    break

            block_start = block_end

        return max_scores

    scores = queries @ corpus.T
    max_scores = torch.empty(
        (num_queries, num_segments),
        device=queries.device,
        dtype=queries.dtype,
    )
    offset = 0
    for idx, count in enumerate(counts):
        segment_scores = scores[:, offset : offset + count]
        max_scores[:, idx] = segment_scores.max(dim=1).values
        offset += count

    return max_scores
