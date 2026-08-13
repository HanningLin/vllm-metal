# SPDX-License-Identifier: Apache-2.0
"""Typed pooling backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.attention.context import OffsetCache

EMBED_TASK: PoolingTask = "embed"
CLASSIFY_TASK: PoolingTask = "classify"


@dataclass(frozen=True, slots=True)
class PoolingCapabilities:
    requires_paged_attention: bool


@dataclass(frozen=True, slots=True)
class DecoderPoolingSpan:
    start_row: int
    num_tokens: int
    is_complete: bool
    pooling_params: PoolingParams


@dataclass(frozen=True, slots=True)
class DecoderPoolingBatch:
    spans: tuple[DecoderPoolingSpan, ...]


class PoolingBackend(Protocol):
    capabilities: PoolingCapabilities

    def supported_tasks(self) -> tuple[PoolingTask, ...]: ...

    def validate_params(self, pooling_params: PoolingParams) -> None: ...


class DecoderPooler(Protocol):
    task: PoolingTask

    def is_supported(self) -> bool: ...

    def validate_params(self, pooling_params: PoolingParams) -> None: ...

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor: ...


class DecoderPoolingBackend(PoolingBackend, Protocol):
    def forward_packed(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache] | None,
    ) -> mx.array: ...

    def pool_packed(
        self,
        hidden_states: mx.array,
        batch: DecoderPoolingBatch,
    ) -> tuple[torch.Tensor | None, ...]: ...
