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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


@dataclass
class BPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`~trl.experimental.bpo.BPOTrainer`].

    Inherits every parameter from [`~trl.GRPOConfig`] and adds the smoothing constant and weight cap from [Bellman
    Policy Optimization](https://huggingface.co/papers/2609.15987). The inherited `epsilon` and `epsilon_high` remain
    the lower and upper clipping bounds, applied to the mismatch-correction weight. BPO always uses
    sequence-mean/token-mean aggregation without a KL penalty or entropy bonus.

    Parameters:
        bpo_smoothing (`float`, *optional*, defaults to `0.1`):
            Additive smoothing constant epsilon in the paper's mismatch-correction weight `(1 + epsilon - mu) / (1 +
            epsilon - pi)`, where `mu` and `pi` are rollout and current token probabilities. This is separate from the
            inherited clipping bounds.
        bpo_weight_cap (`float`, *optional*, defaults to `3.0`):
            Upper bound C on the detached mismatch-correction weight multiplying the token log probability. The
            clipping mask uses the weight before this cap is applied.
    """

    bpo_smoothing: float = field(
        default=0.1,
        metadata={
            "help": "Additive smoothing epsilon in (1 + epsilon - mu) / (1 + epsilon - pi), using token probabilities."
        },
    )
    bpo_weight_cap: float = field(
        default=3.0,
        metadata={"help": "Cap C on the detached BPO mismatch-correction weight, applied after computing the mask."},
    )
