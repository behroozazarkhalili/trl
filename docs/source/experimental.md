# Experimental trainers

See the [experimental overview](experimental_overview) for the stability contract.

## FlashREINFORCE

[`FlashREINFORCETrainer`](flash_reinforce) implements batch-centered, single-rollout REINFORCE with detached token importance sampling, sequence trust, and sample-mean reduction. Import `FlashREINFORCEConfig` and `FlashREINFORCETrainer` from `trl.experimental.flash_reinforce`.

## GMPO

[`GMPOTrainer`](gmpo) implements Geometric-Mean Policy Optimization. Import `GMPOConfig` and `GMPOTrainer` from `trl.experimental.gmpo`.
