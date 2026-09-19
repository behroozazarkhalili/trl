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

import torch

from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id
from .flash_reinforce_config import FlashREINFORCEConfig


class FlashREINFORCETrainer(GRPOTrainer):
    """
    Trainer for the base FlashREINFORCE objective.

    Centers rewards across independent prompts, gates complete trajectories by mean sampled-action Bernoulli KL, and
    averages detached-ratio REINFORCE losses within trajectories and then across the original batch. Generation, weight
    synchronization, and optimization are inherited from [`GRPOTrainer`]; collection is synchronous.

    Uses [`experimental.flash_reinforce.FlashREINFORCEConfig`] with GRPO's model, reward function, and dataset
    arguments. Stored sampling log probabilities take precedence over old policy log probabilities. The standard
    Transformers generation path uses the current detached policy for its single on-policy update and requires
    full-support sampling. Optional failure-token entropy filtering is not implemented.
    """

    _tag_names = ["trl", "flash_reinforce"]

    def __init__(self, model, reward_funcs, args=None, **kwargs):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = FlashREINFORCEConfig(f"{model_name.split('/')[-1]}-FlashREINFORCE")

        if not isinstance(args, FlashREINFORCEConfig):
            raise ValueError("FlashREINFORCETrainer requires FlashREINFORCEConfig.")
        if not args.use_vllm and kwargs.get("rollout_func") is None:
            if (
                args.top_k != 0
                or args.top_p != 1.0
                or args.min_p is not None
                or args.repetition_penalty != 1.0
                or args.generation_kwargs
                or not args.disable_dropout
            ):
                raise ValueError(
                    "FlashREINFORCE's Transformers path requires full-support sampling: top_k=0, top_p=1, "
                    "min_p=None, repetition_penalty=1, no generation_kwargs, and disable_dropout=True. "
                    "Use a rollout_func with stored sampling log probabilities for other behavior policies."
                )

        super().__init__(model, reward_funcs, args=args, **kwargs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        # The parent has already gathered rewards across processes, before any accumulation microbatch split.
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        if torch.isnan(rewards_per_func).all(dim=1).any() or not torch.isfinite(rewards).all():
            raise ValueError("FlashREINFORCE requires a finite aggregate reward for every trajectory.")
        self._batch_advantages = (rewards - rewards.mean()).detach()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        # Replace the parent's single-member group advantages with the complete batch baseline.
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        output["advantages"] = self._batch_advantages[process_slice]
        for _ in range(min(len(self._batch_advantages), len(self._logs["advantages"]))):
            self._logs["advantages"].pop()
        self._logs["advantages"].extend(self._batch_advantages.tolist())
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["frac_reward_zero_std"][-1] = float(self._batch_advantages.eq(0).all())
        return output

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        action_mask = mask.bool()
        lengths = action_mask.sum(-1)
        if (lengths == 0).any():
            raise ValueError("Every FlashREINFORCE trajectory must contain at least one policy token.")
        # Actual sampling probabilities take precedence over a recomputed old-policy forward pass.
        if (self.use_vllm or self.rollout_func is not None) and "sampling_per_token_logps" not in inputs:
            raise ValueError("FlashREINFORCE requires sampling_per_token_logps from vLLM or custom rollouts.")
        behavior_logps = inputs.get("sampling_per_token_logps", inputs.get("old_per_token_logps"))
        if behavior_logps is None:
            behavior_logps = per_token_logps.detach()
        for values in (per_token_logps, behavior_logps):
            valid = values[action_mask]
            if not torch.isfinite(valid).all() or (valid > 0).any():
                raise ValueError("policy-token log probabilities must be finite and <= 0")

        # Reference: flashreinforce/loss.py:57-69,88-95 (base method, no optional failure-token filtering).
        dtype = torch.float64 if per_token_logps.dtype == torch.float64 else torch.float32
        current = torch.where(action_mask, per_token_logps.to(dtype), 0.0)
        with torch.no_grad():
            behavior = torch.where(action_mask, behavior_logps.to(dtype), 0.0)
            advantages = inputs["advantages"].to(dtype)
            p = behavior.exp().clamp(1e-6, 1 - 1e-6)
            q = current.detach().exp().clamp(1e-6, 1 - 1e-6)
            divergence = p * (p.log() - q.log()) + (1 - p) * (torch.log1p(-p) - torch.log1p(-q))
            sequence_kl = torch.where(action_mask, divergence, 0.0).sum(-1) / lengths
            sequence_kl = sequence_kl.clamp_min(0)
            admitted = sequence_kl <= self.args.delta
            active = action_mask & admitted[:, None]
            ratio = torch.where(active, current.detach() - behavior, 0.0).exp()
            if not torch.isfinite(ratio).all():
                raise FloatingPointError("importance ratio overflow on an admitted token")
            weights = torch.where(active, ratio * advantages[:, None], 0.0)

        loss = -((weights * current).sum(-1) / lengths).mean()
        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        loss = loss / normalizer

        self._metrics[mode]["sequence_kl"].append(self.accelerator.gather(sequence_kl).mean().item())
        self._metrics[mode]["acceptance_rate"].append(self.accelerator.gather(admitted.float()).mean().item())

        def global_masked_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": one value per sequence
                local_sum, local_count = x.sum(), torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        return loss
