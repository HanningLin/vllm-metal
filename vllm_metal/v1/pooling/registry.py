# SPDX-License-Identifier: Apache-2.0
"""Pooling backend selection."""

from __future__ import annotations

from typing import Any

from vllm_metal.v1.pooling.backends.decoder.factory import (
    load_decoder_pooling_backend,
)
from vllm_metal.v1.pooling.contract import DecoderPoolingBackend


def load_pooling_backend(
    model: Any,
    model_config: Any,
    tokenizer: Any,
) -> DecoderPoolingBackend:
    return load_decoder_pooling_backend(model, model_config, tokenizer)
