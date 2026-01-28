from __future__ import annotations

import numpy as np
import torch


class ChamferGPU:
    def __init__(self, train: np.ndarray, train_counts: np.ndarray) -> None:
        if train.ndim != 2:
            raise ValueError("train must be a 2D numpy array.")
        if train.dtype != np.float32:
            raise ValueError("train must have dtype float32.")
        if train_counts.ndim != 1:
            raise ValueError("train_counts must be a 1D numpy array.")
        if train_counts.dtype != np.int32:
            raise ValueError("train_counts must have dtype int32.")
        if np.any(train_counts < 0):
            raise ValueError("train_counts must be non-negative.")
        if train_counts.sum(dtype=np.int64) != train.shape[0]:
            raise ValueError("sum(train_counts) must equal the number of rows in train.")

        self.train = torch.from_numpy(train).contiguous().to("cuda")
        offsets = np.concatenate(([0], np.cumsum(train_counts, dtype=np.int64)))
        self._offsets_t = torch.from_numpy(offsets).to("cuda")
        self._num_segments = len(train_counts)

    @torch.inference_mode()
    def query_from_indices(
        self,
        Q: torch.Tensor,
        I: torch.Tensor,
        n: int,
        batch_chunk_size: int = 256,
    ) -> torch.Tensor:
        if Q.ndim != 3 or I.ndim != 2:
            raise ValueError("Q must be 3D and I must be 2D.")
        if Q.device != I.device:
            raise ValueError("Q and I must be on the same device.")

        if batch_chunk_size <= 0:
            raise ValueError("batch_chunk_size must be positive.")
        batch_size, num_queries, dim = Q.shape
        _, num_candidates = I.shape
        device = Q.device

        train_emb = self.train
        offsets = self._offsets_t

        out = torch.empty((batch_size, n), device=device, dtype=I.dtype)

        for b_start in range(0, batch_size, batch_chunk_size):
            b_end = min(b_start + batch_chunk_size, batch_size)
            curr_bs = b_end - b_start

            Q_sub = Q[b_start:b_end]
            I_sub = I[b_start:b_end]

            I_flat = I_sub.reshape(-1)
            seg_starts = offsets.index_select(0, I_flat)
            seg_ends = offsets.index_select(0, I_flat + 1)
            seg_lengths = seg_ends - seg_starts

            batch_lengths = seg_lengths.view(curr_bs, num_candidates).sum(dim=1)
            max_pts = int(batch_lengths.max().item())

            if max_pts == 0:
                out[b_start:b_end] = I_sub[:, :n]
                continue

            total_points = int(seg_lengths.sum().item())

            point_to_seg_idx = torch.repeat_interleave(
                torch.arange(curr_bs * num_candidates, device=device), seg_lengths
            )

            batch_starts_cum = torch.cumsum(batch_lengths, dim=0) - batch_lengths
            batch_ids = point_to_seg_idx.div(num_candidates, rounding_mode="floor")

            seq_arange = torch.arange(total_points, device=device)
            col_indices = seq_arange - batch_starts_cum[batch_ids]

            seg_starts_cum = torch.cumsum(seg_lengths, dim=0) - seg_lengths
            idx_in_seg = seq_arange - seg_starts_cum[point_to_seg_idx]
            src_indices = seg_starts[point_to_seg_idx] + idx_in_seg

            P_dense = torch.empty((curr_bs, max_pts, dim), device=device, dtype=train_emb.dtype)
            P_dense.index_put_((batch_ids, col_indices), train_emb[src_indices])

            scores = torch.bmm(Q_sub, P_dense.transpose(1, 2))

            col_range = torch.arange(max_pts, device=device)
            valid_mask = col_range.unsqueeze(0) < batch_lengths.unsqueeze(1)
            scores.masked_fill_(~valid_mask.unsqueeze(1), -float("inf"))

            cand_map = torch.zeros((curr_bs, max_pts), device=device, dtype=torch.long)
            cand_ids = point_to_seg_idx.remainder(num_candidates)
            cand_map.index_put_((batch_ids, col_indices), cand_ids)

            reduced_scores = torch.full(
                (curr_bs, num_queries, num_candidates),
                -float("inf"),
                device=device,
                dtype=scores.dtype,
            )
            idx_expanded = cand_map.unsqueeze(1).expand(-1, num_queries, -1)
            reduced_scores.scatter_reduce_(
                2, idx_expanded, scores, reduce="amax", include_self=False
            )

            final_scores = reduced_scores.sum(dim=1)
            topk = torch.topk(final_scores, k=n, dim=1).indices
            out[b_start:b_end] = torch.gather(I_sub, 1, topk)

        return out
