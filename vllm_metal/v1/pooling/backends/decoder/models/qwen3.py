# SPDX-License-Identifier: Apache-2.0
"""Qwen3 reranker pooling behavior."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import torch
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.contract import DecoderPoolingSpan
from vllm_metal.v1.pooling.validation import CLASSIFY_POOLER_TASKS, PoolingConfigView


class Qwen3RerankerPooler:
    """Pool Qwen3 reranker hidden states as yes-minus-no scores."""

    task: PoolingTask = "classify"

    def __init__(self, model: Any, model_config: Any, tokenizer: Any) -> None:
        self.model = model
        self.config = PoolingConfigView(model_config)
        self.tokenizer = tokenizer

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        if not self._supported():
            return ()
        return ("classify",)

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if not self._supported():
            raise NotImplementedError(
                "Metal classify pooling requires original Qwen3 reranker "
                "classifier_from_token=['no', 'yes'] and either lm_head for "
                "untied checkpoints or embed_tokens.as_linear for tied "
                "checkpoints."
            )

    def pool_one(
        self,
        hidden_states: mx.array,
        span: DecoderPoolingSpan,
    ) -> torch.Tensor:
        token_index = span.start_row + span.num_tokens - 1
        return self.pool_token(hidden_states, token_index, span.pooling_params)

    def pool_token(
        self,
        hidden_states: mx.array,
        token_index: int,
        pooling_params: PoolingParams,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "Metal classify pooling expected hidden states with shape "
                f"[1, tokens, hidden], got {hidden_states.shape} for model="
                f"{self.config.label}."
            )
        if token_index < 0 or token_index >= hidden_states.shape[1]:
            raise ValueError(
                f"Metal classify pooling token index {token_index} is outside hidden "
                f"state shape {hidden_states.shape} for model={self.config.label}."
            )

        token_ids = self._classifier_token_ids()
        logits_fn = self._classifier_logits_fn()
        if token_ids is None or logits_fn is None:
            raise NotImplementedError(
                "Metal classify pooling requires original Qwen3 reranker "
                "classifier_from_token=['no', 'yes'] and either lm_head for "
                "untied checkpoints or embed_tokens.as_linear for tied "
                "checkpoints."
            )

        no_id, yes_id = token_ids
        vector = hidden_states[0, token_index, :].astype(mx.float32)
        vocab_logits = mx.squeeze(logits_fn(vector).astype(mx.float32))
        if vocab_logits.ndim != 1:
            raise ValueError(
                "Metal classify pooling expected classifier logits with shape "
                f"[vocab], got {vocab_logits.shape} for model={self.config.label}."
            )

        token_logits = vocab_logits[mx.array([no_id, yes_id], dtype=mx.int32)]
        score = token_logits[1] - token_logits[0]
        if self.config.logit_mean is not None:
            score = score - self.config.logit_mean
        if self.config.logit_sigma is not None:
            score = score / self.config.logit_sigma
        if self._classifier_use_activation(pooling_params):
            score = mx.sigmoid(score)

        tensor = mlx_to_torch(score.reshape((1,)), device="cpu")
        return tensor.detach().clone()

    def _supported(self) -> bool:
        if self.config.has_multimodal_config:
            return False
        if self.config.task not in CLASSIFY_POOLER_TASKS:
            return False
        if self.config.unsupported_sequence_pooling_type is not None:
            return False
        if self.config.chunked_processing_enabled:
            return False
        return (
            self.config.is_qwen3_reranker
            and self._classifier_logits_fn() is not None
            and self._classifier_token_ids() is not None
        )

    def _sequence_model(self) -> Any | None:
        inner = getattr(self.model, "model", None)
        return inner if callable(inner) else None

    def _word_embeddings_tied(self) -> bool | None:
        for source in (
            self.model,
            getattr(self.model, "args", None),
            self.config.hf_config,
        ):
            value = getattr(source, "tie_word_embeddings", None)
            if value is not None:
                return bool(value)
        return None

    def _tied_embedding_logits_fn(self) -> Any | None:
        body = self._sequence_model()
        if body is None:
            return None
        embed_tokens = getattr(body, "embed_tokens", None)
        as_linear = getattr(embed_tokens, "as_linear", None)
        return as_linear if callable(as_linear) else None

    def _classifier_logits_fn(self) -> Any | None:
        if self._sequence_model() is None:
            return None

        lm_head = getattr(self.model, "lm_head", None)
        tied_embedding_logits = self._tied_embedding_logits_fn()
        tied = self._word_embeddings_tied()

        if tied is False:
            return lm_head if callable(lm_head) else None
        if tied is True:
            return tied_embedding_logits
        return None

    def _resolve_token_id(self, token: str) -> int | None:
        if self.tokenizer is None:
            return None

        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id

        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            token_ids = encode(token, add_special_tokens=False)
            if isinstance(token_ids, list) and len(token_ids) == 1:
                return int(token_ids[0])

        return None

    def _classifier_token_ids(self) -> tuple[int, int] | None:
        tokens = self.config.classifier_tokens
        if tokens is None:
            return None
        token_ids = tuple(self._resolve_token_id(token) for token in tokens)
        if any(token_id is None for token_id in token_ids):
            return None
        no_id, yes_id = token_ids
        assert no_id is not None and yes_id is not None
        return (no_id, yes_id)

    def _classifier_use_activation(self, pooling_params: PoolingParams) -> bool:
        if pooling_params.use_activation is not None:
            return pooling_params.use_activation
        return self.config.use_activation_by_default
