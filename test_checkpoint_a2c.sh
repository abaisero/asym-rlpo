#!/bin/bash

env=gv_yaml/gv_memory.5x5.yaml
algo=a2c
args=(
  --wandb-project adqn
  --wandb-tag checkpoint-test
  # --wandb-offline
  --max-simulation-timesteps 1_000_000
  --max-episode-timesteps 100
  --evaluation
  --evaluation-period 1
  --checkpoint checkpoint_$algo.pk
)

timeout_time=120
timeout --foreground $timeout_time python ./main_a2c.py $env $algo ${args[@]}
