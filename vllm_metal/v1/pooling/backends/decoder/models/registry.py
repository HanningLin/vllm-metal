# SPDX-License-Identifier: Apache-2.0
"""Model-family poolers for the decoder pooling backend."""

from __future__ import annotations

from typing import Any

from vllm_metal.v1.pooling.backends.decoder.models.qwen3 import Qwen3RerankerPooler
from vllm_metal.v1.pooling.contract import DecoderPooler


def build_decoder_model_poolers(
    model: Any,
    sequence_model: Any | None,
    model_config: Any,
    tokenizer: Any,
) -> tuple[DecoderPooler, ...]:
    return (
        Qwen3RerankerPooler(
            model,
            sequence_model,
            model_config,
            tokenizer,
        ),
    )
