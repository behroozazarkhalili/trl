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
from dataclasses import dataclass, field

from ...trainer.base_config import _BaseConfig
from ...trainer.grpo_config import GRPOConfig


@dataclass
class FlashREINFORCEConfig(GRPOConfig):
    """
    Configuration class for [`experimental.flash_reinforce.FlashREINFORCETrainer`].

    Inherits generation and optimization settings from [`GRPOConfig`]. FlashREINFORCE uses one rollout per prompt,
    batch-centered rewards without standard-deviation scaling, and one optimizer step per generation batch. The base
    objective has no reference KL, entropy bonus, or token filtering. GRPO's `loss_type`, `epsilon`, and `epsilon_high`
    do not select or clip this loss. Liger's GRPO loss is unsupported.

    Parameters:
        delta (`float`, *optional*, defaults to `0.003`):
            Maximum mean sampled-action Bernoulli KL from the behavior policy to the learner for admitting a
            trajectory. Set to `float("inf")` to disable rejection. This replaces GRPO's unrelated ratio cap.
        num_generations (`int`, *optional*, defaults to `1`):
            Number of completions per prompt. Must be `1` for FlashREINFORCE.
        scale_rewards (`str`, *optional*, defaults to `"none"`):
            Must be `"none"`. Rewards are centered over the complete update batch without rescaling.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model. Required for on-policy Transformers generation so that sampling
            and training use the same policy probabilities.
        vllm_importance_sampling_correction (`bool`, *optional*, defaults to `False`):
            Must be `False`. FlashREINFORCE directly uses the stored sampling probabilities for its unclipped token
            importance ratios instead of GRPO's separate clipped correction.
    """

    delta: float = field(default=3e-3, metadata={"help": "Sequence admission threshold for mean Bernoulli KL."})
    num_generations: int = field(default=1, metadata={"help": "One rollout per prompt for FlashREINFORCE."})
    scale_rewards: str = field(
        default="none", metadata={"help": "Center rewards over the full batch without scaling."}
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    vllm_importance_sampling_correction: bool = field(
        default=False, metadata={"help": "Use FlashREINFORCE's own behavior importance correction."}
    )

    def __post_init__(self):
        # Like MiniLLMConfig, bypass GRPO's group-size validation: this method requires single-rollout batches.
        _BaseConfig.__post_init__(self)

        if self.num_generations != 1 or self.num_generations_eval not in (None, 1):
            raise ValueError("FlashREINFORCE requires one generation per prompt in training and evaluation.")
        if self.num_iterations != 1:
            raise ValueError("FlashREINFORCE requires num_iterations=1: each rollout batch is updated once.")
        if self.scale_rewards != "none" or self.multi_objective_aggregation != "sum_then_normalize":
            raise ValueError("FlashREINFORCE requires scale_rewards='none' and sum_then_normalize reward aggregation.")
        if self.delta is None or math.isnan(self.delta) or self.delta < 0:
            raise ValueError("FlashREINFORCE delta must be nonnegative (or infinity to disable sequence rejection).")
        if self.beta != 0 or self.use_liger_kernel or self.vllm_importance_sampling_correction:
            raise ValueError(
                "FlashREINFORCE requires beta=0, use_liger_kernel=False, and no separate vLLM correction."
            )
        if self.entropy_coef != 0 or self.use_adaptive_entropy or self.top_entropy_quantile != 1.0:
            raise ValueError("The base FlashREINFORCE objective does not use entropy bonuses or token filtering.")
        if self.off_policy_mask_threshold is not None or self.mask_truncated_completions:
            raise ValueError(
                "FlashREINFORCE uses its sequence gate without additional off-policy or truncation masks."
            )

        if self.auto_find_batch_size:
            raise ValueError("FlashREINFORCE requires a fixed generation batch; auto_find_batch_size is unsupported.")
        if self.parallelism_config is not None and (
            self.parallelism_config.cp_enabled or self.parallelism_config.sp_enabled
        ):
            raise ValueError("FlashREINFORCE inherits GRPO's restriction against sequence-dimension parallelism.")

        # All microbatches of a generation batch contribute to the same optimizer step.
        generation_batch_size = self.per_device_train_batch_size * self.world_size * self.gradient_accumulation_steps
        if self.generation_batch_size not in (None, generation_batch_size):
            raise ValueError("FlashREINFORCE generation_batch_size must equal the effective optimizer batch size.")
        if self.steps_per_generation not in (None, self.gradient_accumulation_steps):
            raise ValueError("FlashREINFORCE steps_per_generation must equal gradient_accumulation_steps.")
        self.generation_batch_size = generation_batch_size
        self.steps_per_generation = self.gradient_accumulation_steps
        if self.generation_batch_size < 2:
            raise ValueError("FlashREINFORCE needs at least two trajectories per update for batch reward centering.")
