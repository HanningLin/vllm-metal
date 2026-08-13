# SPDX-License-Identifier: Apache-2.0
"""Pooling request and model-config validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.pooling_params import PoolingParams
from vllm.v1.core.sched.output import NewRequestData

from vllm_metal.v1.pooling.contract import PoolingBackend

EMBED_POOLER_TASKS = (None, "embed")
CLASSIFY_POOLER_TASKS = (None, "classify")
QWEN3_RERANKER_TOKENS = ("no", "yes")
SUPPORTED_POOLER_TASKS = EMBED_POOLER_TASKS + ("classify",)
LAST_POOLING = (None, "LAST")


@dataclass(frozen=True, slots=True)
class PoolingConfigView:
    """Small view over vLLM/HF pooling config fields used by Metal."""

    model_config: Any

    @property
    def label(self) -> str:
        served_model_name = getattr(self.model_config, "served_model_name", None)
        if isinstance(served_model_name, (list, tuple)):
            served_model_name = served_model_name[0] if served_model_name else None
        return str(served_model_name or getattr(self.model_config, "model", "unknown"))

    @property
    def hf_config(self) -> Any:
        return getattr(self.model_config, "hf_config", None)

    @property
    def pooler_config(self) -> Any:
        return getattr(self.model_config, "pooler_config", None)

    @property
    def runner_type(self) -> str | None:
        runner_type = getattr(self.model_config, "runner_type", None)
        return str(runner_type) if runner_type is not None else None

    @property
    def task(self) -> str | None:
        task = getattr(self.pooler_config, "task", None)
        return str(task) if task is not None else None

    @property
    def architectures(self) -> tuple[str, ...]:
        architectures: list[str] = []
        for source in (self.model_config, self.hf_config):
            values = getattr(source, "architectures", None)
            if isinstance(values, (list, tuple)):
                architectures.extend(str(value) for value in values)
        return tuple(architectures)

    @property
    def has_multimodal_config(self) -> bool:
        return getattr(self.model_config, "multimodal_config", None) is not None

    @property
    def unsupported_sequence_pooling_type(self) -> str | None:
        for pooling_type in self.sequence_pooling_types:
            if pooling_type not in LAST_POOLING:
                return pooling_type
        return None

    @property
    def sequence_pooling_types(self) -> tuple[str | None, str | None]:
        if self.pooler_config is None:
            return (None, None)
        seq_pooling_type = getattr(self.pooler_config, "seq_pooling_type", None)
        pooling_type = getattr(self.pooler_config, "pooling_type", None)
        return (
            str(seq_pooling_type) if seq_pooling_type is not None else None,
            str(pooling_type) if pooling_type is not None else None,
        )

    @property
    def embed_activation_allowed(self) -> bool:
        if self.pooler_config is None:
            return True
        return getattr(self.pooler_config, "use_activation", None) is not False

    @property
    def chunked_processing_enabled(self) -> bool:
        return bool(getattr(self.pooler_config, "enable_chunked_processing", False))

    @property
    def classifier_tokens(self) -> tuple[str, str] | None:
        tokens = getattr(self.hf_config, "classifier_from_token", None)
        if not isinstance(tokens, (list, tuple)) or len(tokens) != 2:
            return None
        return (str(tokens[0]), str(tokens[1]))

    @property
    def is_decoder_embedding(self) -> bool:
        return any(
            architecture.endswith("ForCausalLM")
            or architecture.endswith("ForTextEncoding")
            or architecture.endswith("EmbeddingModel")
            for architecture in self.architectures
        )

    @property
    def is_qwen3_reranker(self) -> bool:
        return (
            "Qwen3ForSequenceClassification" in self.architectures
            and getattr(self.hf_config, "is_original_qwen3_reranker", False) is True
            and self.classifier_tokens == QWEN3_RERANKER_TOKENS
        )

    @property
    def logit_mean(self) -> float | None:
        value = getattr(self.pooler_config, "logit_mean", None)
        return float(value) if value is not None else None

    @property
    def logit_sigma(self) -> float | None:
        value = getattr(self.pooler_config, "logit_sigma", None)
        return float(value) if value is not None else None

    @property
    def use_activation_by_default(self) -> bool:
        return getattr(self.pooler_config, "use_activation", None) is not False

    @property
    def has_embedding_dimension_override(self) -> bool:
        return getattr(self.pooler_config, "dimensions", None) is not None

    def reject_unsupported_pooler_config(self) -> None:
        if self.task not in SUPPORTED_POOLER_TASKS:
            raise NotImplementedError(
                "Metal pooling supports only pooler_config.task unset, 'embed', "
                f"or 'classify'; got {self.task!r} for model={self.label}."
            )

        sequence_pooling_type = self.unsupported_sequence_pooling_type
        if sequence_pooling_type is not None:
            raise NotImplementedError(
                "Metal pooling currently supports only LAST sequence pooling; "
                f"got {sequence_pooling_type!r} for model={self.label}."
            )
        if self.chunked_processing_enabled:
            raise NotImplementedError(
                "Metal pooling does not support "
                "pooler_config.enable_chunked_processing=True with LAST pooling; "
                f"model={self.label}."
            )

    def unsupported_pooling_option(self, pooling_params: PoolingParams) -> str | None:
        if pooling_params.late_interaction_params is not None:
            return "late-interaction parameters"
        if pooling_params.requires_token_ids:
            return "token-level ALL pooling outputs"
        if pooling_params.step_tag_id is not None:
            return "STEP pooling parameters"
        if pooling_params.returned_token_ids is not None:
            return "returned_token_ids"
        if pooling_params.extra_kwargs:
            return "extra pooling kwargs"
        if pooling_params.task != "classify" and pooling_params.use_activation is False:
            return "use_activation=False"
        if (
            pooling_params.dimensions is not None
            or self.has_embedding_dimension_override
        ):
            return "embedding-dimension truncation"
        return None

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if self.runner_type != "pooling":
            raise NotImplementedError(
                "Metal pooling requires runner_type='pooling'; got "
                f"{self.runner_type!r} for model={self.label}."
            )
        self.reject_unsupported_pooler_config()

        task = pooling_params.task
        if task in EMBED_POOLER_TASKS:
            if not self.is_decoder_embedding:
                raise NotImplementedError(
                    "Metal embed pooling requires a decoder-style checkpoint; got "
                    f"architectures={self.architectures!r} for model={self.label}."
                )
        elif task == "classify":
            if not self.is_qwen3_reranker:
                raise NotImplementedError(
                    "Metal classify pooling currently supports only original Qwen3 "
                    "reranker checkpoints converted with "
                    "Qwen3ForSequenceClassification and classifier_from_token="
                    "['no', 'yes']; "
                    f"architectures={self.architectures!r} for model={self.label}."
                )
        else:
            raise NotImplementedError(
                "Metal pooling supports only text-only task='embed' and the "
                "Qwen3 reranker task='classify' for now; "
                f"got task={task!r} for model={self.label}."
            )

        unsupported_option = self.unsupported_pooling_option(pooling_params)
        if unsupported_option is not None:
            raise NotImplementedError(
                f"Metal pooling does not support {unsupported_option} "
                f"for model={self.label}."
            )


def validate_pooling_params(
    pooling_params: PoolingParams,
    model_config: Any,
) -> None:
    PoolingConfigView(model_config).validate_params(pooling_params)


def validate_pooling_request(
    new_req: NewRequestData,
    model_config: Any,
    backend: PoolingBackend | None,
    paged_attention_enabled: bool,
) -> None:
    pooling_params = new_req.pooling_params
    if pooling_params is None:
        return

    if backend is None:
        raise RuntimeError("Metal pooling backend is not installed.")

    PoolingConfigView(model_config).validate_params(pooling_params)
    backend.validate_params(pooling_params)
    if new_req.mm_features:
        raise NotImplementedError(
            "Multimodal pooling inputs are not supported on Metal yet."
        )
    if new_req.prompt_embeds is not None:
        raise NotImplementedError(
            "Prompt-embedding pooling inputs are not supported on Metal yet."
        )
    if backend.capabilities.requires_paged_attention and not paged_attention_enabled:
        raise NotImplementedError(
            "Metal pooling currently requires paged attention; "
            "set VLLM_METAL_USE_PAGED_ATTENTION=1."
        )
    if not (new_req.prompt_token_ids or []):
        raise ValueError(
            f"Metal pooling requires prompt_token_ids for request {new_req.req_id!r}."
        )
