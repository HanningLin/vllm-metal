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
        self._contiguous_runs: tuple[tuple[int | None, int, int], ...] = ()
        self._token_indices_by_slot: tuple[tuple[int, mx.array], ...] = ()
        self._no_lora = True

    @property
    def no_lora(self) -> bool:
        return self._no_lora

    def update_metadata(
        self, mapping: LoRAMapping, lora_index_to_id: list[int | None]
    ) -> None:
        slot_of = {aid: i for i, aid in enumerate(lora_index_to_id) if aid is not None}
        runs: list[tuple[int | None, int, int]] = []
        token_indices_by_slot: dict[int, list[int]] = {}
        run_slot: int | None = None
        run_start = 0
        for token_index, adapter_id in enumerate(mapping.index_mapping):
            slot = slot_of.get(adapter_id)
            if slot is not None:
                token_indices_by_slot.setdefault(slot, []).append(token_index)
            if token_index == 0:
                run_slot = slot
                continue
            if slot != run_slot:
                runs.append((run_slot, run_start, token_index))
                run_slot = slot
                run_start = token_index
        if mapping.index_mapping:
            runs.append((run_slot, run_start, len(mapping.index_mapping)))

        active_slot_count = len(token_indices_by_slot)
        use_contiguous_runs = active_slot_count > 0 and len(runs) <= active_slot_count
        self._contiguous_runs = tuple(runs) if use_contiguous_runs else ()
        self._token_indices_by_slot = (
            ()
            if use_contiguous_runs
            else tuple(
                (slot, mx.array(indices, dtype=mx.int32))
                for slot, indices in sorted(token_indices_by_slot.items())
            )
        )
        self._no_lora = active_slot_count == 0

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

        if self._contiguous_runs:
            outputs: list[mx.array] = []
            for slot, start, end in self._contiguous_runs:
                y_run = y[start:end]
                if slot is None:
                    outputs.append(y_run)
                    continue
                rank = ranks[slot]
                if rank == 0:
                    outputs.append(y_run)
                    continue
                lora_a = lora_a_stacked[slot, :rank]
                lora_b = lora_b_stacked[slot, :, :rank]
                x_run = x[start:end]
                delta = mx.matmul(mx.matmul(x_run, lora_a.T), lora_b.T)
                outputs.append(y_run + delta * scale)
            return mx.concatenate(outputs, axis=0)

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
