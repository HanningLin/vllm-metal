# SPDX-License-Identifier: Apache-2.0
"""MLX PunicaWrapper — grouped matmuls for the rank-r LoRA delta.

No-LoRA tokens are passed through without indexing a weight slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from vllm.lora.layers import LoRAMapping


class PunicaWrapperMLX:
    def __init__(self, max_num_batched_tokens: int, max_batches: int, max_loras: int):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_batches = max_batches
        self.max_loras = max_loras
        self._token_indices_by_slot: tuple[tuple[int, mx.array], ...] = ()
        self._no_lora = True

    @property
    def no_lora(self) -> bool:
        return self._no_lora

    def update_metadata(
        self, mapping: LoRAMapping, lora_index_to_id: list[int | None]
    ) -> None:
        slot_of = {aid: i for i, aid in enumerate(lora_index_to_id) if aid is not None}
        token_indices_by_slot: dict[int, list[int]] = {}
        for token_index, adapter_id in enumerate(mapping.index_mapping):
            slot = slot_of.get(adapter_id)
            if slot is not None:
                token_indices_by_slot.setdefault(slot, []).append(token_index)
        self._token_indices_by_slot = tuple(
            (slot, mx.array(indices, dtype=mx.int32))
            for slot, indices in sorted(token_indices_by_slot.items())
        )
        self._no_lora = not self._token_indices_by_slot

    def add_lora_linear(
        self,
        y: mx.array,
        x: mx.array,
        lora_a_stacked: mx.array,
        lora_b_stacked: mx.array,
        scale: float,
        lora_ranks: list[int] | None = None,
    ) -> mx.array:
        """Apply LoRA deltas once per active adapter slot."""
        if self._no_lora:
            return y
        max_rank = int(lora_a_stacked.shape[1])
        ranks = lora_ranks or [max_rank] * self.max_loras
        output = y
        for slot, token_indices in self._token_indices_by_slot:
            rank = ranks[slot]
            if rank == 0:
                continue
            lora_a = lora_a_stacked[slot, :rank]
            lora_b = lora_b_stacked[slot, :, :rank]
            x_for_slot = mx.take(x, token_indices, axis=0)
            delta = mx.matmul(mx.matmul(x_for_slot, lora_a.T), lora_b.T)
            output = output.at[token_indices].add(delta * scale)
        return output
