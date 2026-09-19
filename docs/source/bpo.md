# BPO

[Bellman Policy Optimization](https://huggingface.co/papers/2609.15987) by Zhuoqing Song, Haotian Xu,
Xikun Zhang, and Lidong Bing is a critic-free method derived from Policy Mirror Descent using the Bellman equations.
It keeps GRPO's group-normalized advantages and replaces the importance-sampling ratio with a smoothed ratio of
complementary token probabilities.

Use [`experimental.bpo.BPOTrainer`] and [`experimental.bpo.BPOConfig`] from `trl.experimental.bpo`.
Generation, reward computation, and weight syncing are inherited from [`GRPOTrainer`].

## Usage

```python
from datasets import load_dataset

from trl.experimental.bpo import BPOConfig, BPOTrainer

dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")


def reward_func(completions, **kwargs):
    return [float(len(completion)) for completion in completions]


training_args = BPOConfig(
    output_dir="Qwen3-0.6B-BPO",
    bpo_smoothing=0.1,
    bpo_weight_cap=3.0,
    beta=0.0,
    entropy_coef=0.0,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
)
trainer = BPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```

The length reward is a minimal usage example. For reasoning tasks, supply a verifier that scores response correctness.
Always pass a `BPOConfig` as `args`: the inherited constructor otherwise creates a `GRPOConfig`.

## Loss

For response advantage \\(\hat A^i\\), current policy \\(\pi\\), and rollout policy \\(\mu\\), the per-token loss is

$$
L^{\mathrm{BPO}}_{i,t}(\pi) =
-\hat A^i M_t^i \min\bigl(\operatorname{sg}(\omega_t^i), C\bigr)
\log\pi(y_t^i \mid x, y_{<t}^i),
\qquad
\omega_t^i = \frac{1 + \epsilon - \mu(y_t^i \mid x, y_{<t}^i)}
{1 + \epsilon - \pi(y_t^i \mid x, y_{<t}^i)}.
$$

Both policy values in the weight are token probabilities obtained by exponentiating log probabilities.
`bpo_smoothing` is the paper's \\(\epsilon\\), with default `0.1`; `bpo_weight_cap` is \\(C\\), with default `3.0`.
The weight is detached, so the gradient flows only through the final log probability.

The mask is zero when either of these conditions holds, and one otherwise:

- The advantage is positive and the uncapped weight exceeds `1 + epsilon_high`.
- The advantage is negative and the uncapped weight is below `1 - epsilon`.

Here `epsilon` and `epsilon_high` are the inherited GRPO clipping fields, distinct from `bpo_smoothing`.
`epsilon` defaults to `0.2`; `epsilon_high=None` uses the same value. The paper does not explicitly give BPO's
clipping bounds, so these inherited defaults are retained. The mask is evaluated before applying the cap.

BPO averages over valid tokens within each response, then over responses, as in Appendix B.
Padding and tool-output tokens are excluded; clipped tokens remain in the denominator with zero loss.
There is no KL penalty or entropy bonus. Entropy and clipping fractions are logged using the GRPO metric names.

When rollout-engine log probabilities are available, they define \\(\mu\\). Otherwise BPO uses cached old-policy
log probabilities, or detached current log probabilities for GRPO's on-policy shortcut. Missing engine log
probabilities marked as NaN use the cached policy at those positions. To match the paper's sampling setup, use
temperature `1.0`, top-p `1.0`, and top-k `0`.

This experimental variant fixes the loss to the paper's objective. Inherited GRPO options for alternative losses,
sequence importance sampling, additional off-policy or entropy filtering, KL and entropy regularization, and MoE
auxiliary losses do not change this objective. No additional vLLM importance-sampling multiplier is applied.
Keep those options at their defaults when using BPO; `loss_type` does not select its aggregation.

## BPOTrainer

[[autodoc]] experimental.bpo.BPOTrainer
    - train
    - save_model
    - push_to_hub

## BPOConfig

[[autodoc]] experimental.bpo.BPOConfig
