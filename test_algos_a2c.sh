#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

envs=(PO-pos-CartPole-v1 ../gym-gridverse/yaml/gv_empty.4x4.yaml)
algos=(sym-a2c asym-a2c)

args=(
  --max-simulation-timesteps 500
)

for env in ${envs[@]}; do
  for algo in ${algos[@]}; do
    echo ./main_a2c.py $env $algo ${args[@]}
    python -W ignore ./main_a2c.py $env $algo ${args[@]} > /dev/null
    [ $? -eq 0 ] && echo "SUCCESS" || echo "FAIL"
  done
done

exit 0
