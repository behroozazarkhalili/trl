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
from ...trainer.utils import nanmax, nanmin


class BPOTrainer(GRPOTrainer):
    """
    Trainer for Bellman Policy Optimization (BPO).

    [BPO](https://huggingface.co/papers/2609.15987) replaces GRPO's importance-sampling ratio with a smoothed ratio of
    complementary token probabilities. The clipped, capped weight is detached and multiplies `log pi` directly.

    The only change w.r.t. [`~trl.GRPOTrainer`] is `_compute_loss`. Everything else (generation, reward computation,
    weight syncing, metric logging) is inherited unchanged. Pass a [`~trl.experimental.bpo.BPOConfig`] as `args`. The
    loss always uses sequence-mean/token-mean aggregation without a KL penalty or entropy bonus.
    """

    _tag_names = ["trl", "bpo"]
    _name = "BPO"
    _paper = {"title": "Bellman Policy Optimization", "id": "2609.15987"}

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
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

        advantages = inputs["advantages"].unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps, so we skip its computation and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # BPO uses the behavior policy, including rollout-engine discrepancies when sampling logprobs are available.
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
        # GRPO marks unavailable rollout logprobs as NaN; use the cached policy for those tokens.
        sampling_per_token_logps = torch.where(
            sampling_per_token_logps.isnan(), old_per_token_logps, sampling_per_token_logps
        )
        omega = (
            (1 + self.args.bpo_smoothing - sampling_per_token_logps.exp())
            / (1 + self.args.bpo_smoothing - per_token_logps.exp())
        ).detach()

        # Apply the sign-dependent mask to the uncapped weight, then cap its detached magnitude (paper, Eq. 14).
        is_low_clipped = (omega < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (omega > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        weights = torch.clamp(omega, max=self.args.bpo_weight_cap)
        per_token_loss = -advantages * ~is_region_clipped * weights * per_token_logps

        # BPO uses sequence-mean/token-mean aggregation (paper, Appendix B).
        mode = "train" if self.model.training else "eval"
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        loss = loss / normalizer

        # Log the metrics
        def masked_seq_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": already one value per sequence
                return x.squeeze(1)
            return (x * mask).sum(-1) / mask.sum(-1)

        def global_masked_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": one value per sequence
                local_sum, local_count = x.sum(), torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))
        self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/high_mean"].append(global_masked_mean(is_high_clipped.float()))
        self._metrics[mode]["clip_ratio/region_mean"].append(global_masked_mean(is_region_clipped.float()))
        gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(masked_seq_mean(is_high_clipped.float()))
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())

        return loss
