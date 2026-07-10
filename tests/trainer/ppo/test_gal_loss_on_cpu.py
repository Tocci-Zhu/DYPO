# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from verl.trainer.ppo.core_algos import (
    compute_gal_pairwise_loss_and_advantages,
    compute_gal_surrogate_loss,
)


def test_gal_surrogate_matches_exact_pairwise_gradient():
    """The decomposed surrogate must preserve GAL for an asymmetric group."""
    beta = 0.7
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    anchor_log_probs = torch.tensor(
        [
            [-0.3, -0.7, -9.0],
            [-0.4, -0.2, -0.5],
            [-1.2, -9.0, -9.0],
            [-0.9, -0.8, -9.0],
            [-0.6, -0.7, -0.9],
        ],
        requires_grad=True,
    )
    ref_log_probs = torch.tensor(
        [
            [-0.5, -0.8, -8.0],
            [-0.6, -0.5, -0.4],
            [-1.0, -8.0, -8.0],
            [-0.8, -0.9, -8.0],
            [-0.7, -0.6, -0.8],
        ]
    )

    anchor_sequence_log_probs = (anchor_log_probs * response_mask).sum(dim=-1)
    ref_sequence_log_probs = (ref_log_probs * response_mask).sum(dim=-1)
    pairwise_losses, success_advantages, failure_advantages = (
        compute_gal_pairwise_loss_and_advantages(
            success_log_probs=anchor_sequence_log_probs[:2],
            failure_log_probs=anchor_sequence_log_probs[2:],
            beta=beta,
            success_ref_log_probs=ref_sequence_log_probs[:2],
            failure_ref_log_probs=ref_sequence_log_probs[2:],
        )
    )
    exact_loss = pairwise_losses.mean()
    exact_gradient = torch.autograd.grad(exact_loss, anchor_log_probs)[0]

    gal_advantages = torch.cat((success_advantages, failure_advantages))
    assert torch.allclose(gal_advantages.sum(), torch.tensor(0.0), atol=1e-6)

    current_log_probs = torch.nn.Parameter(anchor_log_probs.detach().clone())
    surrogate_loss = compute_gal_surrogate_loss(
        current_log_probs=current_log_probs,
        response_mask=response_mask,
        gal_loss_values=exact_loss.detach().expand(5),
        gal_advantages=gal_advantages,
        gal_mask=torch.ones(5, dtype=torch.bool),
    )
    surrogate_gradient = torch.autograd.grad(surrogate_loss, current_log_probs)[0]

    assert surrogate_loss.requires_grad
    assert torch.allclose(surrogate_loss.detach(), exact_loss.detach())
    assert torch.allclose(surrogate_gradient, exact_gradient, atol=1e-6, rtol=1e-6)
    assert torch.all(surrogate_gradient[:2][response_mask[:2].bool()] < 0)
    assert torch.all(surrogate_gradient[2:][response_mask[2:].bool()] > 0)
    assert torch.equal(
        surrogate_gradient[response_mask == 0],
        torch.zeros_like(surrogate_gradient[response_mask == 0]),
    )


def test_gal_gradient_survives_single_sample_micro_batches():
    """Gradient accumulation with micro-batch size one must retain GAL."""
    beta = 0.2
    coef = 0.7
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    anchor_log_probs = torch.zeros((2, 3), requires_grad=True)
    sequence_log_probs = (anchor_log_probs * response_mask).sum(dim=-1)
    pairwise_losses, success_advantages, failure_advantages = (
        compute_gal_pairwise_loss_and_advantages(
            success_log_probs=sequence_log_probs[:1],
            failure_log_probs=sequence_log_probs[1:],
            beta=beta,
        )
    )
    exact_gradient = torch.autograd.grad(
        coef * pairwise_losses.mean(), anchor_log_probs
    )[0]

    current_log_probs = torch.nn.Parameter(anchor_log_probs.detach().clone())
    gal_advantages = torch.cat((success_advantages, failure_advantages))
    accumulated_loss = current_log_probs.sum() * 0.0
    for sample_idx in range(2):
        micro_batch_loss = compute_gal_surrogate_loss(
            current_log_probs=current_log_probs[sample_idx : sample_idx + 1],
            response_mask=response_mask[sample_idx : sample_idx + 1],
            gal_loss_values=pairwise_losses.mean().detach().reshape(1),
            gal_advantages=gal_advantages[sample_idx : sample_idx + 1],
            gal_mask=torch.ones(1, dtype=torch.bool),
        )
        accumulated_loss = accumulated_loss + coef * micro_batch_loss / 2

    accumulated_loss.backward()
    accumulated_gradient = current_log_probs.grad
    expected_gradient = torch.tensor(
        [
            [-coef * beta / 2, -coef * beta / 2, 0.0],
            [coef * beta / 2, coef * beta / 2, 0.0],
        ]
    )

    assert accumulated_loss.requires_grad
    assert accumulated_gradient is not None
    assert torch.count_nonzero(accumulated_gradient) > 0
    assert torch.allclose(accumulated_gradient, exact_gradient, atol=1e-7, rtol=1e-7)
    assert torch.allclose(accumulated_gradient, expected_gradient, atol=1e-7, rtol=1e-7)
