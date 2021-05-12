#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

envs=(POMDP-heavenhell-episodic-v0 PO-pos-CartPole-v1 gv_yaml/gv_four_rooms.7x7.yaml)
algos=(sym-a2c asym-a2c)

args=(
  --max-simulation-timesteps 500
  --max-episode-timesteps 100
)

for env in ${envs[@]}; do
  for algo in ${algos[@]}; do
    echo ./main_a2c.py $env $algo ${args[@]}
    python -W ignore ./main_a2c.py $env $algo ${args[@]} > /dev/null
    [ $? -eq 0 ] && echo "SUCCESS" || echo "FAIL"
  done
done

exit 0
