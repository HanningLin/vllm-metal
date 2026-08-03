# SPDX-License-Identifier: Apache-2.0
"""Focused tests for paged-prefill sampling orchestration."""

from types import SimpleNamespace

import numpy as np
import torch
from vllm.sampling_params import SamplingParams

import vllm_metal.v1.sampling_batch as sampling_batch


def test_prefill_sampling_batches_scheduler_order_and_request_metadata(
    monkeypatch,
) -> None:
    """Multiple final prefills cross the sampler boundary once as one batch."""
    logits = np.zeros((1, 6, 4), dtype=np.float16)
    logits[0, 3, 1] = 5
    logits[0, 5, 2] = 7
    first_params = SamplingParams(temperature=0.0, logprobs=1)
    second_params = SamplingParams(temperature=0.7, top_k=3, logprobs=1)
    second_generator = torch.Generator().manual_seed(42)
    prefill_reqs = [
        SimpleNamespace(
            token_ids=[12],
            full_prompt_token_ids=[10, 11, 12],
            prompt_len=3,
            sampling_params=first_params,
            generator=None,
        ),
        SimpleNamespace(
            token_ids=[30, 31, 32],
            full_prompt_token_ids=None,
            prompt_len=3,
            sampling_params=second_params,
            generator=second_generator,
        ),
    ]
    sampler_calls: list[tuple[np.ndarray, sampling_batch.SamplingBatch]] = []

    def fake_sample_from_logits(
        logits_2d: np.ndarray,
        batch: sampling_batch.SamplingBatch,
        sampler: object,
        device: torch.device,
    ) -> sampling_batch._SamplingResult:
        del sampler, device
        sampler_calls.append((logits_2d, batch))
        token_ids = np.argmax(logits_2d, axis=-1).astype(np.int32)
        logprob_token_ids = np.stack((token_ids, np.zeros_like(token_ids)), axis=1)
        logprobs = np.stack(
            (-token_ids.astype(np.float32), -token_ids.astype(np.float32) - 1),
            axis=1,
        )
        return sampling_batch._SamplingResult(
            token_ids.tolist(),
            sampling_batch.LogprobsLists(
                logprob_token_ids=logprob_token_ids,
                logprobs=logprobs,
                sampled_token_ranks=token_ids,
            ),
        )

    monkeypatch.setattr(sampling_batch.mx, "stack", np.stack)
    monkeypatch.setattr(sampling_batch, "sample_from_logits", fake_sample_from_logits)

    result = sampling_batch.sample_prefill_tokens(
        logits,
        prefill_reqs,
        cu_seqlens=[0, 2, 4, 6],
        num_decode=1,
        sampler=object(),
        device=torch.device("cpu"),
        vocab_size=4,
        logitsprocs=SimpleNamespace(),
    )

    assert len(sampler_calls) == 1
    sampled_logits, batch = sampler_calls[0]
    np.testing.assert_array_equal(sampled_logits, logits[0, [3, 5], :])
    assert sampled_logits.dtype == logits.dtype
    assert batch.sampling_params_list == [first_params, second_params]
    assert batch.prompt_token_id_lists == [[10, 11, 12], [30, 31, 32]]
    assert batch.output_token_id_lists == [[], []]
    assert batch.generators == {1: second_generator}
    assert result.token_ids == [1, 2]
    assert result.logprobs is not None
    np.testing.assert_array_equal(result.logprobs.logprob_token_ids[:, 0], [1, 2])
    np.testing.assert_array_equal(result.logprobs.sampled_token_ranks, [1, 2])


def test_prefill_sampling_empty_batch_skips_sampler(monkeypatch) -> None:
    """An empty prefill batch preserves the no-work result."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("empty prefill sampling must not invoke the sampler")

    monkeypatch.setattr(sampling_batch.mx, "stack", fail)
    monkeypatch.setattr(sampling_batch, "sample_from_logits", fail)

    result = sampling_batch.sample_prefill_tokens(
        np.zeros((1, 0, 4), dtype=np.float16),
        [],
        cu_seqlens=[0],
        num_decode=0,
        sampler=object(),
        device=torch.device("cpu"),
        vocab_size=4,
        logitsprocs=SimpleNamespace(),
    )

    assert result == sampling_batch._SamplingResult([])
