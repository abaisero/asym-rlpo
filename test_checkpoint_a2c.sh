#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

env=gv_yaml/gv_memory.5x5.yaml
algo=a2c
args=(
  --wandb-project adqn
  --wandb-tag checkpoint-test
  --wandb-offline
  --max-simulation-timesteps 1_000_000
  --max-episode-timesteps 100
  --evaluation
  --evaluation-period 100
  --run-path "./tmp-checkpoint-test/"
  --checkpoint-period 10
)

timeout --foreground 120s python ./main_a2c.py $env $algo ${args[@]}
# python ./main_a2c.py $env $algo ${args[@]}
