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
from collections import defaultdict, deque
from unittest.mock import patch

import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.flash_reinforce import FlashREINFORCEConfig, FlashREINFORCETrainer

from ..testing_utils import TrlTestCase


@pytest.fixture
def loss_case():
    trainer = FlashREINFORCETrainer.__new__(FlashREINFORCETrainer)
    trainer.args = GRPOConfig("dummy", bf16=False, loss_type="grpo", report_to="none")
    trainer.model = torch.nn.Linear(1, 1)
    trainer.accelerator = Accelerator(cpu=True)
    trainer.current_gradient_accumulation_steps = 1
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.top_entropy_quantile = 1.0
    trainer.aux_loss_enabled = False
    trainer._entropy_bonus_enabled = False
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = "token"
    trainer.loss_type = "grpo"
    trainer.beta = 0.0
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.use_vllm = False
    trainer.rollout_func = None
    trainer.args.delta = 0.003

    logps = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -2.0, -3.0], [-12.0, -13.0, -99.0]], requires_grad=True)
    behavior = torch.tensor([[-4.0, -1.0, -1.0], [-1.02, -2.01, -3.01], [-13.0, -14.0, -99.0]], requires_grad=True)
    rewards = torch.tensor([3.0, 0.0, 1.0], requires_grad=True)
    inputs = {
        "prompt_ids": torch.ones(3, 1, dtype=torch.long),
        "prompt_mask": torch.ones(3, 1, dtype=torch.long),
        "completion_ids": torch.ones(3, 3, dtype=torch.long),
        "completion_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "tool_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1]]),
        "old_per_token_logps": behavior,
        "advantages": rewards - rewards.mean(),
    }
    with patch.object(
        trainer, "_get_per_token_logps_and_entropies", return_value=(logps, torch.ones_like(logps), None)
    ):
        yield trainer, inputs, logps, behavior, rewards


class TestFlashREINFORCELoss:
    def test_loss_and_gradient_match_numeric_oracle(self, loss_case):
        trainer, inputs, logps, behavior, rewards = loss_case
        mask = (inputs["completion_mask"] * inputs["tool_mask"]).bool()
        expected = torch.tensor(0.0)
        expected_grad = torch.zeros_like(logps)
        admitted = []
        # Independent, per-row calculation of the reference loss, including unequal lengths and a rejected row.
        for i in range(3):
            current = logps[i, mask[i]].detach()
            old = behavior[i, mask[i]].detach()
            p, q = old.exp().clamp(1e-6, 1 - 1e-6), current.exp().clamp(1e-6, 1 - 1e-6)
            kl = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()
            admitted.append(kl <= 0.003)
            if admitted[-1]:
                advantage = torch.tensor([5 / 3, -4 / 3, -1 / 3])[i]
                weights = torch.exp(current - old) * advantage
                expected -= (weights * current).mean() / 3
                expected_grad[i, mask[i]] = -weights / (3 * mask[i].sum())
        assert admitted == [False, True, True]

        loss = trainer._compute_loss(trainer.model, inputs)
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-6)
        loss.backward()
        torch.testing.assert_close(logps.grad, expected_grad, rtol=0, atol=1e-6)
        assert behavior.grad is None
        assert rewards.grad is None

    def test_sequence_trust_changes_loss_and_differs_from_grpo(self, loss_case):
        trainer, inputs, _, _, _ = loss_case
        loss = trainer._compute_loss(trainer.model, inputs)
        # Removing the sequence gate must fail this assertion even if detached-ratio REINFORCE is retained.
        trainer.args.delta = math.inf
        ungated_loss = trainer._compute_loss(trainer.model, inputs)
        assert not torch.isclose(loss, ungated_loss, atol=1e-6)
        trainer.args.delta = None  # GRPO's unrelated optional ratio cap
        grpo_loss = GRPOTrainer._compute_loss(trainer, trainer.model, inputs)
        assert not torch.isclose(loss, grpo_loss, atol=1e-6)

    def test_sampling_probabilities_take_precedence(self, loss_case):
        trainer, inputs, logps, behavior, _ = loss_case
        expected = trainer._compute_loss(trainer.model, inputs)
        inputs["sampling_per_token_logps"] = behavior
        inputs["old_per_token_logps"] = logps.detach()
        loss = trainer._compute_loss(trainer.model, inputs)
        torch.testing.assert_close(loss, expected)

    def test_accumulation_scaling_and_evaluation(self, loss_case):
        trainer, inputs, _, _, _ = loss_case
        expected = trainer._compute_loss(trainer.model, inputs)
        trainer.current_gradient_accumulation_steps = 3
        torch.testing.assert_close(trainer._compute_loss(trainer.model, inputs), expected / 3)
        trainer.model.eval()
        torch.testing.assert_close(trainer._compute_loss(trainer.model, inputs), expected)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.1])
    def test_invalid_behavior_probabilities_are_rejected(self, loss_case, value):
        trainer, inputs, _, behavior, _ = loss_case
        with torch.no_grad():
            behavior[1, 0] = value
        with pytest.raises(ValueError, match="log probabilities"):
            trainer._compute_loss(trainer.model, inputs)

    def test_external_rollouts_require_sampling_probabilities(self, loss_case):
        trainer, inputs, _, _, _ = loss_case
        trainer.use_vllm = True
        with pytest.raises(ValueError, match="sampling_per_token_logps"):
            trainer._compute_loss(trainer.model, inputs)

    def test_masked_nonfinite_values_do_not_affect_loss_or_gradients(self, loss_case):
        trainer, inputs, logps, behavior, _ = loss_case
        expected = trainer._compute_loss(trainer.model, inputs).detach()
        with torch.no_grad():
            logps[1, 2] = torch.nan
            behavior[1, 2] = torch.inf
            logps[0, 2] = torch.inf  # tool observation, excluded even though completion_mask is one
            behavior[0, 2] = torch.nan
        loss = trainer._compute_loss(trainer.model, inputs)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        assert torch.isfinite(logps.grad).all()
        assert logps.grad[:, 2].eq(0).all()

    @pytest.mark.parametrize("all_rejected", [False, True])
    def test_zero_gradient_for_constant_rewards_or_all_rejected(self, loss_case, all_rejected):
        trainer, inputs, logps, _, _ = loss_case
        if all_rejected:
            inputs["old_per_token_logps"] = torch.full_like(logps, -0.1)
        else:
            inputs["advantages"] = torch.zeros(3)
        loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        assert loss.item() == 0
        assert logps.grad.eq(0).all()


class TestFlashREINFORCEConfig:
    def test_single_rollout_with_odd_batch_size(self):
        args = FlashREINFORCEConfig("dummy", per_device_train_batch_size=3, gradient_accumulation_steps=3, bf16=False)
        assert args.num_generations == 1
        assert args.generation_batch_size == 9
        assert args.steps_per_generation == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_generations": 2},
            {"num_iterations": 2},
            {"steps_per_generation": 2},
            {"generation_batch_size": 16},
            {"scale_rewards": "group"},
            {"beta": 0.1},
            {"use_liger_kernel": True},
            {"auto_find_batch_size": True},
        ],
    )
    def test_rejects_settings_that_change_the_update_rule(self, kwargs):
        with pytest.raises(ValueError, match="FlashREINFORCE"):
            FlashREINFORCEConfig("dummy", bf16=False, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"top_k": 10},
            {"top_p": 0.9},
            {"min_p": 0.1},
            {"repetition_penalty": 1.1},
            {"generation_kwargs": {"do_sample": False}},
            {"disable_dropout": False},
        ],
    )
    def test_transformers_generation_requires_matching_behavior_distribution(self, kwargs):
        args = FlashREINFORCEConfig("dummy", bf16=False, **kwargs)
        with pytest.raises(ValueError, match="full-support"):
            FlashREINFORCETrainer(model="unused", reward_funcs=[], args=args)


class TestFlashREINFORCETrainer(TrlTestCase):
    def test_evaluation_keeps_previous_completion_log_advantages(self, loss_case):
        trainer, _, _, _, _ = loss_case
        trainer.model.eval()
        trainer._logs = {"advantages": deque([-2.0, -1.0, 0.0, 0.0], maxlen=4)}
        trainer._metrics["eval"]["frac_reward_zero_std"].append(1.0)
        trainer._batch_advantages = torch.tensor([0.5, -0.5])
        with patch.object(GRPOTrainer, "_generate_and_score_completions", return_value={}):
            output = trainer._generate_and_score_completions([{}, {}])
        assert list(trainer._logs["advantages"]) == [-2.0, -1.0, 0.5, -0.5]
        torch.testing.assert_close(output["advantages"], torch.tensor([0.5, -0.5]))

    def test_train(self):
        dataset = Dataset.from_dict(
            {"prompt": ["Count:", "Write:", "Explain:", "Say:"], "score": [0.0, 1.0, 2.0, 3.0]}
        )

        def reward_func(score, **kwargs):
            return score

        training_args = FlashREINFORCEConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.01,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_completion_length=4,
            max_steps=2,
            bf16=False,
            use_cpu=True,
            gradient_checkpointing=False,
            save_strategy="no",
            report_to="none",
        )
        trainer = FlashREINFORCETrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        previous_trainable_params = {n: param.detach().clone() for n, param in trainer.model.named_parameters()}
        result = trainer.train()

        assert trainer.state.global_step == 2
        assert math.isfinite(result.training_loss)
        assert any(
            not torch.equal(param, trainer.model.get_parameter(n)) for n, param in previous_trainable_params.items()
        )
        # One generation per distinct prompt; advantages use the full update batch, not each microbatch.
        assert trainer.num_generations == 1
        assert sorted(trainer._logs["advantages"]) == [-1.5, -0.5, 0.5, 1.5]
