# SPDX-License-Identifier: Apache-2.0
"""Typed pooling backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.attention.context import OffsetCache


class PoolingExecutionKind(Enum):
    DECODER = "decoder"
    ENCODER = "encoder"


@dataclass(frozen=True, slots=True)
class PoolingCapabilities:
    execution_kind: PoolingExecutionKind
    requires_paged_attention: bool
    uses_kv_cache: bool
    supports_chunked_requests: bool


@dataclass(frozen=True, slots=True)
class DecoderPoolingSpan:
    req_id: str
    start_row: int
    num_tokens: int
    is_complete: bool
    pooling_params: PoolingParams


@dataclass(frozen=True, slots=True)
class DecoderPoolingBatch:
    spans: tuple[DecoderPoolingSpan, ...]


@dataclass(frozen=True, slots=True)
class PoolingBatchResult:
    outputs: tuple[torch.Tensor | None, ...]


class PoolingBackend(Protocol):
    capabilities: PoolingCapabilities

    def supported_tasks(self) -> tuple[PoolingTask, ...]: ...

    def validate_params(self, pooling_params: PoolingParams) -> None: ...


class DecoderPooler(Protocol):
    task: PoolingTask

    def supported_tasks(self) -> tuple[PoolingTask, ...]: ...

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
    ) -> PoolingBatchResult: ...
