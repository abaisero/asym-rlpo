#!/bin/bash

export WANDB_MODE=dryrun

envs=(PO-pos-CartPole-v1 ../gym-gridverse/yaml/gv_empty.4x4.yaml)
algos=(fob-dqn foe-dqn poe-dqn poe-adqn)

args=(
  --max-simulation-timesteps 11_000
)

for env in ${envs[@]}; do
  for algo in ${algos[@]}; do
    echo ./main_dqn.py $env $algo ${args[@]}
    ./main_dqn.py $env $algo ${args[@]} &> /dev/null
    [[ $? ]] && echo "SUCCESS" || echo "FAIL"
  done
done

exit 0
