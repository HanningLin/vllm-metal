# SPDX-License-Identifier: Apache-2.0
"""Decoder pooling batch helpers used by the Metal runner."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.v1.pooling.contract import (
    DecoderPoolingBackend,
    DecoderPoolingBatch,
    DecoderPoolingSpan,
    PoolingBatchResult,
)


def has_paged_pooling_work(
    prefill_reqs: list[Any],
    decode_reqs: list[Any],
) -> bool:
    """Return whether a paged batch is pure pooling work."""
    pooling_prefills = [pr for pr in prefill_reqs if pr.pooling_params is not None]
    has_pooling_work = bool(pooling_prefills)
    if has_pooling_work and (len(pooling_prefills) != len(prefill_reqs) or decode_reqs):
        raise NotImplementedError(
            "Metal pooling batches cannot mix pooling requests with "
            "generation prefill/decode requests."
        )
    return has_pooling_work


def build_decoder_pooling_batch(
    prefill_entries: list[Any],
    cu_seqlens: list[int],
    num_decode_segments: int,
) -> DecoderPoolingBatch:
    spans: list[DecoderPoolingSpan] = []
    for index, entry in enumerate(prefill_entries):
        pooling_params = entry.prefill.pooling_params
        if pooling_params is None:
            raise RuntimeError(
                "Paged pooling batch contained a non-pooling prefill request."
            )
        start = cu_seqlens[num_decode_segments + index]
        end = cu_seqlens[num_decode_segments + index + 1]
        spans.append(
            DecoderPoolingSpan(
                req_id=entry.prefill.req_id,
                start_row=start,
                num_tokens=end - start,
                is_complete=entry.result_mode != "intermediate",
                pooling_params=pooling_params,
            )
        )
    return DecoderPoolingBatch(tuple(spans))


def pool_paged_prefill_batch(
    backend: DecoderPoolingBackend,
    hidden_states: mx.array,
    prefill_entries: list[Any],
    cu_seqlens: list[int],
    num_decode_segments: int,
) -> PoolingBatchResult:
    """Pool paged decoder prefill outputs in scheduler order."""
    mx.eval(hidden_states)
    batch = build_decoder_pooling_batch(
        prefill_entries,
        cu_seqlens,
        num_decode_segments,
    )
    return backend.pool_packed(hidden_states, batch)
