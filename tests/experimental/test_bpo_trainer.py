# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import math
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.bpo import BPOConfig, BPOTrainer

from ..testing_utils import TrlTestCase


def bpo_reference(logps, old_logps, advantages, mask, smoothing, cap, epsilon_low, epsilon_high):
    # Independent scalar evaluation of paper equations (14), (15), and the clipping mask.
    # Python floats and math.exp use double precision; no trainer helpers or tensor loss operations are used.
    loss = 0.0
    gradients = []
    for i in range(len(logps)):
        sequence_loss = 0.0
        sequence_gradients = []
        length = max(sum(mask[i]), 1)
        for t in range(len(logps[i])):
            pi = math.exp(logps[i][t])
            mu = math.exp(old_logps[i][t])
            omega = (1 + smoothing - mu) / (1 + smoothing - pi)
            active = 1.0
            if advantages[i] > 0 and omega > 1 + epsilon_high:
                active = 0.0
            if advantages[i] < 0 and omega < 1 - epsilon_low:
                active = 0.0
            weight = min(omega, cap)
            coefficient = -advantages[i] * active * weight * mask[i][t]
            sequence_loss += coefficient * logps[i][t]
            sequence_gradients.append(coefficient / length / len(logps))
        loss += sequence_loss / length / len(logps)
        gradients.append(sequence_gradients)
    return loss, gradients


def make_loss_inputs(probabilities, old_probabilities, advantages):
    logps = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    batch_size, length = logps.shape
    inputs = {
        "prompt_ids": torch.zeros((batch_size, 1), dtype=torch.long),
        "prompt_mask": torch.ones((batch_size, 1), dtype=torch.long),
        "completion_ids": torch.zeros((batch_size, length), dtype=torch.long),
        "completion_mask": torch.ones((batch_size, length), dtype=torch.long),
        "advantages": torch.tensor(advantages, dtype=torch.float64),
    }
    if old_probabilities is not None:
        inputs["old_per_token_logps"] = torch.tensor(old_probabilities, dtype=torch.float64).log().requires_grad_()
    return logps, inputs


def make_loss_trainer(logps, training=True, accumulation_steps=1, smoothing=0.1, cap=3.0):
    # Exercise the actual loss with controlled model outputs, without loading a language model for each numeric case.
    trainer = object.__new__(BPOTrainer)
    trainer.args = SimpleNamespace(bpo_smoothing=smoothing, bpo_weight_cap=cap)
    trainer.model = SimpleNamespace(training=training)
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.28
    trainer.current_gradient_accumulation_steps = accumulation_steps
    trainer.accelerator = SimpleNamespace(reduce=lambda tensor, reduction: tensor, gather=lambda tensor: tensor)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._get_per_token_logps_and_entropies = Mock(return_value=(logps, torch.zeros_like(logps), None))
    return trainer


class TestBPOConfig:
    def test_defaults_preserve_grpo_clipping(self):
        args = BPOConfig("dummy", bf16=False)
        assert args.bpo_smoothing == 0.1
        assert args.bpo_weight_cap == 3.0
        assert args.epsilon == 0.2
        assert args.epsilon_high is None
        assert args.beta == 0.0
        assert args.entropy_coef == 0.0


class TestBPOLoss:
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    @pytest.mark.parametrize("accumulation_steps", [1, 3])
    def test_numeric_oracle(self, training, accumulation_steps):
        logps, inputs = make_loss_inputs(
            [[0.8, 0.3, 0.6, 0.4], [0.1, 0.95, 0.6, 0.2], [0.4, 0.7, 0.2, 0.8]],
            [[0.1, 0.4, 0.5, 0.7], [0.9, 0.1, 0.5, 0.3], [0.5, 0.6, 0.9, 0.4]],
            [1.0, -1.0, 0.0],
        )
        inputs["completion_mask"] = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        inputs["tool_mask"] = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1]])
        trainer = make_loss_trainer(logps, training=training, accumulation_steps=accumulation_steps)
        expected, expected_gradients = bpo_reference(
            logps.tolist(),
            inputs["old_per_token_logps"].tolist(),
            inputs["advantages"].tolist(),
            (inputs["completion_mask"] * inputs["tool_mask"]).tolist(),
            smoothing=0.1,
            cap=3.0,
            epsilon_low=0.2,
            epsilon_high=0.28,
        )
        normalizer = accumulation_steps if training else 1
        loss = trainer._compute_loss(trainer.model, inputs)
        torch.testing.assert_close(loss, torch.tensor(expected / normalizer, dtype=torch.float64), rtol=0, atol=1e-14)
        loss.backward()
        torch.testing.assert_close(
            logps.grad, torch.tensor(expected_gradients, dtype=torch.float64) / normalizer, rtol=0, atol=1e-14
        )
        assert inputs["old_per_token_logps"].grad is None

    @pytest.mark.parametrize("pi, mu, advantage", [(0.9, 0.1, 1.0), (0.1, 0.9, -1.0)])
    def test_clipped_tokens_contribute_exactly_zero(self, pi, mu, advantage):
        logps, inputs = make_loss_inputs([[pi]], [[mu]], [advantage])
        trainer = make_loss_trainer(logps)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        assert loss.item() == 0.0
        assert logps.grad.item() == 0.0
        assert trainer._metrics["train"]["clip_ratio/region_mean"] == [1.0]

    @pytest.mark.parametrize("cap", [2.0, 3.0, 4.0])
    def test_cap_saturates_without_masking_negative_advantage(self, cap):
        logps, inputs = make_loss_inputs([[0.95]], [[0.1]], [-2.0])
        trainer = make_loss_trainer(logps, cap=cap)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        # omega = 1 / 0.15 > C; a negative advantage keeps this token active with weight exactly C.
        assert loss.item() == pytest.approx(2 * cap * math.log(0.95), abs=1e-14)
        assert logps.grad.item() == 2 * cap

    def test_mask_uses_uncapped_weight(self):
        logps, inputs = make_loss_inputs([[0.9]], [[0.1]], [1.0])
        trainer = make_loss_trainer(logps, cap=1.1)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        # The raw weight is 5, above 1.28, even though the capped weight is below 1.28.
        assert loss.item() == 0.0
        assert logps.grad.item() == 0.0

    @pytest.mark.parametrize("advantage", [1.0, -1.0])
    def test_clipping_boundary_is_inclusive(self, advantage):
        logps, inputs = make_loss_inputs([[0.625]], [[0.5 if advantage > 0 else 0.75]], [advantage])
        trainer = make_loss_trainer(logps, smoothing=0.125)
        trainer.epsilon_low = trainer.epsilon_high = 0.25
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        weight = 1.25 if advantage > 0 else 0.75
        assert loss.item() == pytest.approx(-advantage * weight * math.log(0.625), abs=1e-14)
        assert logps.grad.item() == -advantage * weight

    def test_weight_is_directly_detached(self):
        logps, inputs = make_loss_inputs([[0.4]], [[0.3]], [1.0])
        trainer = make_loss_trainer(logps)
        weights = []
        original_clamp = torch.clamp

        def capture_weight(tensor, **kwargs):
            weight = original_clamp(tensor, **kwargs)
            weights.append(weight)
            return weight

        with patch("torch.clamp", side_effect=capture_weight):
            loss = trainer._compute_loss(trainer.model, inputs)

        # Inspect the actual multiplier used by the loss, including its autograd metadata.
        assert len(weights) == 1
        assert not weights[0].requires_grad
        assert weights[0].grad_fn is None
        loss.backward()
        assert inputs["old_per_token_logps"].grad is None
        assert logps.grad.item() == pytest.approx(-8 / 7, abs=1e-14)

    def test_sampling_logps_take_precedence_over_cached_training_logps(self):
        logps, inputs = make_loss_inputs([[0.4]], [[0.99]], [1.0])
        inputs["sampling_per_token_logps"] = torch.tensor([[0.3]], dtype=torch.float64).log().requires_grad_()
        inputs["importance_sampling_ratio"] = torch.zeros_like(logps)
        trainer = make_loss_trainer(logps)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        assert loss.item() == pytest.approx(-8 / 7 * math.log(0.4), abs=1e-14)
        assert logps.grad.item() == pytest.approx(-8 / 7, abs=1e-14)
        assert inputs["sampling_per_token_logps"].grad is None
        assert inputs["old_per_token_logps"].grad is None

    def test_on_policy_without_cached_logps(self):
        logps, inputs = make_loss_inputs([[0.4]], None, [1.0])
        trainer = make_loss_trainer(logps)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        assert loss.item() == pytest.approx(-math.log(0.4), abs=1e-14)
        assert logps.grad.item() == -1.0

    @pytest.mark.parametrize("mask_name", ["completion_mask", "tool_mask"])
    def test_unavailable_sampling_logps_preserve_old_policy_correction(self, mask_name):
        logps, inputs = make_loss_inputs([[0.4, 0.5]], [[0.3, 0.35]], [1.0])
        inputs["sampling_per_token_logps"] = torch.full_like(logps, float("nan"))
        inputs[mask_name] = torch.tensor([[1, 0]])
        trainer = make_loss_trainer(logps)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        # Missing engine logprobs arrive as NaN from GRPO. Preserve the known old/current policy mismatch.
        assert loss.item() == pytest.approx(-8 / 7 * math.log(0.4), abs=1e-14)
        torch.testing.assert_close(logps.grad, torch.tensor([[-8 / 7, 0.0]], dtype=torch.float64))
        assert inputs["old_per_token_logps"].grad is None

    @pytest.mark.parametrize("mu, weight", [(0.1, 3.0), (1.0, 1.0)])
    def test_smoothing_is_finite_at_probability_one(self, mu, weight):
        logps, inputs = make_loss_inputs([[1.0]], [[mu]], [-1.0])
        trainer = make_loss_trainer(logps)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        assert loss.item() == 0.0
        assert logps.grad.item() == weight

    def test_empty_completion_mask(self):
        logps, inputs = make_loss_inputs([[0.4, 0.3]], [[0.4, 0.3]], [1.0])
        inputs["completion_mask"].zero_()
        trainer = make_loss_trainer(logps)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        assert loss.item() == 0.0
        assert torch.equal(logps.grad, torch.zeros_like(logps))


class TestBPOTrainer(TrlTestCase):
    def test_train(self):
        dataset = Dataset.from_dict({"prompt": ["The sky is", "Two plus two equals", "The capital of France is"]})

        def reward_func(completion_ids, **kwargs):
            return [float(sum(ids) % 7) for ids in completion_ids]

        training_args = BPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,
            max_steps=2,
            bf16=False,
            use_cpu=True,
            gradient_checkpointing=False,
            save_strategy="no",
            logging_steps=1,
            report_to="none",
        )
        trainer = BPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            processing_class=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"),
        )
        previous_trainable_params = {n: param.detach().clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.global_step == 2
        assert math.isfinite(trainer.state.log_history[-1]["train_loss"])
        assert any(log.get("grad_norm", 0) > 0 for log in trainer.state.log_history)
        assert any(
            not torch.equal(param, trainer.model.get_parameter(n)) for n, param in previous_trainable_params.items()
        )
