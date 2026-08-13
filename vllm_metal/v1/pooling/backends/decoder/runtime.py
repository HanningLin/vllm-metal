# SPDX-License-Identifier: Apache-2.0
"""Generic decoder pooling backend.

This module owns packed decoder execution and LAST-token embedding pooling.
Model-family files provide only task-specific poolers, for example Qwen3
reranker ``classify`` scoring.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.attention.context import OffsetCache
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.backends.decoder.models.qwen3 import Qwen3RerankerPooler
from vllm_metal.v1.pooling.contract import (
    EMBED_TASK,
    DecoderPooler,
    DecoderPoolingBatch,
    DecoderPoolingSpan,
    PoolingCapabilities,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView

EMBED_POOLER_TASKS = (None, EMBED_TASK)


class DecoderModelView:
    """Access the MLX decoder model shape used by pooling."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.sequence_model = self._sequence_model()

    @property
    def has_sequence_model(self) -> bool:
        return self.sequence_model is not None

    def forward_packed(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache] | None,
    ) -> mx.array:
        assert self.sequence_model is not None
        if offset_caches is None:
            return self.sequence_model(input_ids)
        return self.sequence_model(input_ids, cache=offset_caches)

    def _sequence_model(self) -> Any | None:
        body = getattr(self.model, "model", None)
        return body if callable(body) else None


class LastTokenEmbeddingPooler:
    """Pool decoder hidden states into normalized LAST-token embeddings."""

    task: PoolingTask = EMBED_TASK

    def __init__(
        self,
        model_view: DecoderModelView,
        config: PoolingConfigView,
    ) -> None:
        self.model_view = model_view
        self.config = config

    def is_supported(self) -> bool:
        return (
            self.config.is_text_only
            and self.config.task in EMBED_POOLER_TASKS
            and self.config.uses_last_pooling
            and self.config.embed_activation_allowed
            and not self.config.chunked_processing_enabled
            and self.model_view.has_sequence_model
            and self._is_decoder_embedding()
        )

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if not self.is_supported():
            raise NotImplementedError(
                "Metal embed pooling requires a decoder-style checkpoint; got "
                f"model={self.config.label}."
            )

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        token_index = span.start_row + span.num_tokens - 1
        return self.pool_token(hidden_states, token_index)

    def pool_token(self, hidden_states: mx.array, token_index: int) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "Metal embed pooling expected hidden states with shape "
                f"[1, tokens, hidden], got {hidden_states.shape} "
                f"for model={self.config.label}."
            )
        if token_index < 0 or token_index >= hidden_states.shape[1]:
            raise ValueError(
                f"Metal embed pooling token index {token_index} is outside hidden "
                f"state shape {hidden_states.shape} for model={self.config.label}."
            )

        vector = hidden_states[0, token_index, :].astype(mx.float32)
        vector = self._normalize_vector(vector)
        tensor = mlx_to_torch(vector, device="cpu", already_contiguous=True)
        return tensor.detach().clone()

    def _normalize_vector(self, vector: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(vector * vector))
        norm = mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
        return mx.contiguous(vector / norm)

    def _is_decoder_embedding(self) -> bool:
        return any(
            architecture.endswith("ForCausalLM")
            or architecture.endswith("ForTextEncoding")
            or architecture.endswith("EmbeddingModel")
            for architecture in self.config.architectures
        )


class MetalDecoderPoolingBackend:
    """Decoder pooling backend for current Metal text pooling behavior."""

    capabilities = PoolingCapabilities(requires_paged_attention=True)

    def __init__(self, model: Any, model_config: Any, tokenizer: Any) -> None:
        self.config = PoolingConfigView(model_config)
        self.model_view = DecoderModelView(model)
        self.poolers: tuple[DecoderPooler, ...] = (
            LastTokenEmbeddingPooler(self.model_view, self.config),
            Qwen3RerankerPooler(
                model,
                self.model_view.sequence_model,
                model_config,
                tokenizer,
            ),
        )

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        return tuple(pooler.task for pooler in self.poolers if pooler.is_supported())

    def validate_params(self, pooling_params: PoolingParams) -> None:
        task = pooling_params.task or EMBED_TASK
        for pooler in self.poolers:
            if task == pooler.task:
                pooler.validate_params(pooling_params)
                return
        raise NotImplementedError(
            "Metal pooling supports only text-only task='embed' and the "
            "Qwen3 reranker task='classify' for now; "
            f"got task={pooling_params.task!r} for model="
            f"{self.config.label}."
        )

    def forward_packed(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache] | None,
    ) -> mx.array:
        self.config.reject_unsupported_pooler_config()
        if not self.model_view.has_sequence_model:
            raise NotImplementedError(
                "Metal pooling requires an MLX model with a callable "
                f"'.model' transformer body; model={self.config.label}; "
                "runner='pooling'."
            )
        return self.model_view.forward_packed(input_ids, offset_caches)

    def pool_packed(
        self,
        hidden_states: mx.array,
        batch: DecoderPoolingBatch,
    ) -> tuple[torch.Tensor | None, ...]:
        outputs: list[torch.Tensor | None] = []
        for span in batch.spans:
            if not span.is_complete:
                outputs.append(None)
                continue
            outputs.append(self._pool_complete_span(hidden_states, span))
        return tuple(outputs)

    def _pool_complete_span(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        task = span.pooling_params.task or EMBED_TASK
        for pooler in self.poolers:
            if task == pooler.task and pooler.is_supported():
                return pooler.pool_one(hidden_states, span)
        raise NotImplementedError(
            "Metal pooling supports only text-only task='embed' and the "
            "Qwen3 reranker task='classify' for now; "
            f"got task={span.pooling_params.task!r} for model="
            f"{self.config.label}."
        )


def build_decoder_pooling_backend(
    model: Any,
    model_config: Any,
    tokenizer: Any,
) -> MetalDecoderPoolingBackend:
    return MetalDecoderPoolingBackend(model, model_config, tokenizer)
