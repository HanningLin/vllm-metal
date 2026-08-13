# SPDX-License-Identifier: Apache-2.0
"""Pooling request and model-config validation."""

from __future__ import annotations

from typing import Any

from vllm.pooling_params import PoolingParams
from vllm.v1.core.sched.output import NewRequestData

from vllm_metal.v1.pooling.contract import PoolingBackend

EMBED_POOLER_TASKS = (None, "embed")
CLASSIFY_POOLER_TASKS = (None, "classify")
SUPPORTED_POOLER_TASKS = (None, "embed", "classify")
LAST_POOLING = (None, "LAST")
QWEN3_RERANKER_TOKENS = ("no", "yes")


def model_label(model_config: Any) -> str:
    served = getattr(model_config, "served_model_name", None)
    if isinstance(served, (list, tuple)):
        served = served[0] if served else None
    return str(served or getattr(model_config, "model", "unknown"))


def hf_config(model_config: Any) -> Any:
    return getattr(model_config, "hf_config", None)


def architecture_names(model_config: Any) -> tuple[str, ...]:
    candidates: list[str] = []
    config = hf_config(model_config)
    for source in (model_config, config):
        architectures = getattr(source, "architectures", None)
        if isinstance(architectures, (list, tuple)):
            candidates.extend(str(arch) for arch in architectures)
    return tuple(candidates)


def pooler_config(model_config: Any) -> Any:
    return getattr(model_config, "pooler_config", None)


def pooler_task(model_config: Any) -> str | None:
    task = getattr(pooler_config(model_config), "task", None)
    return str(task) if task is not None else None


def sequence_pooling_types(model_config: Any) -> tuple[str | None, str | None]:
    config = pooler_config(model_config)
    if config is None:
        return (None, None)
    seq_pooling_type = getattr(config, "seq_pooling_type", None)
    pooling_type = getattr(config, "pooling_type", None)
    return (
        str(seq_pooling_type) if seq_pooling_type is not None else None,
        str(pooling_type) if pooling_type is not None else None,
    )


def unsupported_sequence_pooling_type(model_config: Any) -> str | None:
    for pooling_type in sequence_pooling_types(model_config):
        if pooling_type not in LAST_POOLING:
            return pooling_type
    return None


def pooler_activation_allows_embed(model_config: Any) -> bool:
    config = pooler_config(model_config)
    if config is None:
        return True
    return getattr(config, "use_activation", None) is not False


def chunked_processing_enabled(model_config: Any) -> bool:
    return bool(
        getattr(pooler_config(model_config), "enable_chunked_processing", False)
    )


def classifier_tokens(model_config: Any) -> tuple[str, str] | None:
    tokens = getattr(hf_config(model_config), "classifier_from_token", None)
    if not isinstance(tokens, (list, tuple)) or len(tokens) != 2:
        return None
    return (str(tokens[0]), str(tokens[1]))


def reject_unsupported_pooler_config(model_config: Any) -> None:
    task = pooler_task(model_config)
    if task not in SUPPORTED_POOLER_TASKS:
        raise NotImplementedError(
            "Metal pooling supports only pooler_config.task unset, 'embed', "
            f"or 'classify'; got {task!r} for model={model_label(model_config)}."
        )

    seq_pooling_type = unsupported_sequence_pooling_type(model_config)
    if seq_pooling_type is not None:
        raise NotImplementedError(
            "Metal pooling currently supports only LAST sequence pooling; "
            f"got {seq_pooling_type!r} for model={model_label(model_config)}."
        )
    if chunked_processing_enabled(model_config):
        raise NotImplementedError(
            "Metal pooling does not support "
            "pooler_config.enable_chunked_processing=True with LAST pooling; "
            f"model={model_label(model_config)}."
        )


def is_decoder_embedding_config(model_config: Any) -> bool:
    return any(
        arch.endswith("ForCausalLM")
        or arch.endswith("ForTextEncoding")
        or arch.endswith("EmbeddingModel")
        for arch in architecture_names(model_config)
    )


def is_qwen3_token_logit_classifier(model_config: Any) -> bool:
    return (
        "Qwen3ForSequenceClassification" in architecture_names(model_config)
        and getattr(hf_config(model_config), "is_original_qwen3_reranker", False)
        is True
        and classifier_tokens(model_config) == QWEN3_RERANKER_TOKENS
    )


def unsupported_pooling_option(
    pooling_params: PoolingParams,
    model_config: Any,
) -> str | None:
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
        or getattr(pooler_config(model_config), "dimensions", None) is not None
    ):
        return "embedding-dimension truncation"
    return None


def validate_pooling_params(
    pooling_params: PoolingParams,
    model_config: Any,
) -> None:
    model = model_label(model_config)
    if getattr(model_config, "runner_type", None) != "pooling":
        raise NotImplementedError(
            "Metal pooling requires runner_type='pooling'; got "
            f"{getattr(model_config, 'runner_type', None)!r} for model={model}."
        )
    reject_unsupported_pooler_config(model_config)

    task = pooling_params.task
    if task in (None, "embed"):
        if not is_decoder_embedding_config(model_config):
            raise NotImplementedError(
                "Metal embed pooling requires a decoder-style checkpoint; got "
                f"architectures={architecture_names(model_config)!r} for model="
                f"{model}."
            )
    elif task == "classify":
        if not is_qwen3_token_logit_classifier(model_config):
            raise NotImplementedError(
                "Metal classify pooling currently supports only original Qwen3 "
                "reranker checkpoints converted with "
                "Qwen3ForSequenceClassification and classifier_from_token="
                "['no', 'yes']; "
                f"architectures={architecture_names(model_config)!r} for model="
                f"{model}."
            )
    else:
        raise NotImplementedError(
            "Metal pooling supports only text-only task='embed' and the "
            "Qwen3 reranker task='classify' for now; "
            f"got task={task!r} for model={model}."
        )

    unsupported_option = unsupported_pooling_option(pooling_params, model_config)
    if unsupported_option is not None:
        raise NotImplementedError(
            f"Metal pooling does not support {unsupported_option} for model={model}."
        )


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

    validate_pooling_params(pooling_params, model_config)
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
