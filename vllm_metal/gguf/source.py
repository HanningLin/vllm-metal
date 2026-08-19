# SPDX-License-Identifier: Apache-2.0
"""Resolved GGUF load identity carried from vLLM config to the loader."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from huggingface_hub import snapshot_download

_GGUF_SUFFIX = ".gguf"
_REMOTE_REF_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*:[A-Za-z0-9_+-]+$"
)
_QUANT_TAG_RE = re.compile(
    r"(?:^|-)(?:I?Q\d[A-Za-z0-9_]*|F16|F32|BF16|MXFP\d[A-Za-z0-9_]*)$",
    re.IGNORECASE,
)
_REMOTE_PREFIXES = ("*.", "*-")
_REMOTE_SUFFIXES = ("-*", "")
_REMOTE_SHARD_RE = re.compile(r"-\d+-of-\d+\.gguf$")
_SUPPORTED_REMOTE_QTYPES = frozenset({"Q8_0", "Q4_0", "Q4_1"})


@dataclass(frozen=True, slots=True)
class RemoteGGUFReference:
    """Hugging Face ``repo_id:quant`` GGUF weight reference."""

    repo_id: str
    quant_type: str

    @classmethod
    def parse(cls, value: str) -> Self | None:
        if not _REMOTE_REF_RE.fullmatch(value):
            return None
        repo_id, quant_type = value.rsplit(":", 1)
        if _QUANT_TAG_RE.search(quant_type) is None:
            return None
        return cls(repo_id=repo_id, quant_type=quant_type)

    @property
    def value(self) -> str:
        return f"{self.repo_id}:{self.quant_type}"

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        return tuple(
            f"{prefix}{normalized_quant}{suffix}{_GGUF_SUFFIX}"
            for normalized_quant in (self.quant_type.upper(), self.quant_type.lower())
            for prefix, suffix in itertools.product(_REMOTE_PREFIXES, _REMOTE_SUFFIXES)
        )

    def resolve(
        self,
        *,
        cache_dir: str | None,
        revision: str | None,
        ignore_patterns: list[str] | str | None,
    ) -> str:
        if self.quant_type.upper() not in _SUPPORTED_REMOTE_QTYPES:
            supported = ", ".join(sorted(_SUPPORTED_REMOTE_QTYPES))
            raise ValueError(
                f"Remote GGUF qtype {self.quant_type!r} is not supported by "
                f"vllm-metal; supported qtypes: {supported}."
            )
        snapshot_dir = Path(
            snapshot_download(
                repo_id=self.repo_id,
                cache_dir=cache_dir,
                allow_patterns=list(self.allow_patterns),
                revision=revision,
                ignore_patterns=ignore_patterns,
            )
        )
        return str(self._select_single_file(snapshot_dir))

    def _select_single_file(self, snapshot_dir: Path) -> Path:
        matches = sorted(
            {
                match
                for pattern in self.allow_patterns
                for match in snapshot_dir.glob(pattern)
                if match.is_file()
            }
        )
        if not matches:
            raise ValueError(
                f"No {self.quant_type!r} GGUF file found in remote repository "
                f"{self.repo_id!r}."
            )
        if any(_REMOTE_SHARD_RE.search(match.name) for match in matches):
            raise ValueError(
                f"Remote sharded GGUF files are not supported yet: {self.value!r}."
            )
        if len(matches) != 1:
            names = ", ".join(match.name for match in matches)
            raise ValueError(
                f"Remote GGUF reference {self.value!r} matched multiple files: {names}."
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class GGUFLoadSource:
    """Local GGUF weights plus companion config/tokenizer sources."""

    weights_path: str
    config_dir: str
    tokenizer_dir: str

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        load_config: Any | None = None,
    ) -> GGUFLoadSource | None:
        if model_config.quantization != "gguf":
            return None

        weights_ref = model_config.model_weights
        if cls.is_weights_path(weights_ref):
            weights_path = weights_ref
        elif remote_ref := RemoteGGUFReference.parse(weights_ref):
            weights_path = remote_ref.resolve(
                cache_dir=None if load_config is None else load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=(
                    None if load_config is None else load_config.ignore_patterns
                ),
            )
        else:
            raise ValueError(
                "GGUF model_config must carry a local .gguf path or remote "
                f"repo_id:quant reference in model_weights; got {weights_ref!r}."
            )

        config_dir = model_config.model
        return cls(
            weights_path=weights_path,
            config_dir=config_dir,
            tokenizer_dir=model_config.tokenizer or config_dir,
        )

    @staticmethod
    def is_weights_path(value: str) -> bool:
        return value.endswith(_GGUF_SUFFIX)
