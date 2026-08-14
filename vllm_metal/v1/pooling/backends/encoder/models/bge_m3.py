# SPDX-License-Identifier: Apache-2.0
"""BGE-M3 dense encoder pooling."""

from __future__ import annotations

from typing import Any

from vllm_metal.v1.pooling.backends.encoder.models.xlm_roberta import (
    load_xlm_roberta_backend,
)
from vllm_metal.v1.pooling.contract import EncoderPoolingBackend

_ARCHITECTURES = frozenset({"BgeM3EmbeddingModel"})


def supports_bge_m3_encoder(model_config: Any) -> bool:
    architectures = tuple(
        str(value) for value in model_config.hf_config.architectures or ()
    )
    return any(architecture in _ARCHITECTURES for architecture in architectures)


def load_bge_m3_backend(
    model_config: Any,
) -> tuple[Any, Any, dict[str, Any], EncoderPoolingBackend]:
    # Dense BGE-M3 uses the RoBERTa/XLM-R backbone and normal CLS embed pooler.
    # Sparse lexical heads are added separately with token_classify support.
    return load_xlm_roberta_backend(model_config)
