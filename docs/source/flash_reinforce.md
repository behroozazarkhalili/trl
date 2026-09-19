# FlashREINFORCE

FlashREINFORCE centers rewards across a batch of independent prompts, applies detached token importance ratios, and rejects complete trajectories whose mean sampled-action Bernoulli KL exceeds a threshold. Its loss averages over each trajectory's policy tokens and then over the original batch, including rejected trajectories in the denominator.

To use FlashREINFORCE, use [`experimental.flash_reinforce.FlashREINFORCETrainer`] in `trl.experimental.flash_reinforce`. This implements the base objective (`negative_token_fraction=1`) from the [reference implementation](https://github.com/yifanzhang-pro/FlashREINFORCE).

## Usage

```python
from trl.experimental.flash_reinforce import FlashREINFORCEConfig, FlashREINFORCETrainer

training_args = FlashREINFORCEConfig(
    delta=3e-3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
)
trainer = FlashREINFORCETrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

There is one rollout per prompt and one optimizer update per rollout batch. Rewards are centered over the complete batch across devices before splitting into gradient accumulation microbatches, with no standard-deviation scaling. Each trajectory must have a finite aggregate reward and at least one policy token. Prompt, padding, and tool-observation tokens are excluded. Reference-model KL, PPO clipping, and entropy filtering are absent from this objective.

`delta` is the threshold for the mean Bernoulli KL from the behavior policy to the learner, with probabilities clamped to `[1e-6, 1 - 1e-6]` for this gate only. Set `delta=float("inf")` to disable rejection. Importance ratios remain unclipped; their gradients, the reward baseline, and the admission mask are detached.

The trainer inherits GRPO's synchronous generation. It requires stored `sampling_per_token_logps` from vLLM or a custom `rollout_func`; these must describe the actual behavior distribution. Ordinary Transformers generation uses the current detached log probabilities for the single on-policy update. That path requires full-support sampling (`top_k=0`, `top_p=1`, `min_p=None`, `repetition_penalty=1`), `disable_dropout=True` (the default), and no `generation_kwargs` overrides. An asynchronous rollout scheduler and the optional failure-token entropy filter from the source repository are not included.

## FlashREINFORCETrainer

[[autodoc]] experimental.flash_reinforce.FlashREINFORCETrainer
    - train
    - save_model
    - push_to_hub

## FlashREINFORCEConfig

[[autodoc]] experimental.flash_reinforce.FlashREINFORCEConfig
