#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

envs=(POMDP-heavenhell-episodic-v0 PO-pos-CartPole-v1 gv_yaml/gv_four_rooms.7x7.yaml)
algos=(fob-dqn foe-dqn dqn adqn adqn-bootstrap adqn-state adqn-state-bootstrap)

args=(
  --episode-buffer-prepopulate-timesteps 100
  --max-simulation-timesteps 500
  --max-episode-timesteps 100
  # --truncated-histories
  # --truncated-histories-n 10
)

[[ "$#" -eq 0 ]] && echo "running without standard output" || echo "running with standard output"
echo

for env in ${envs[@]}; do
  for algo in ${algos[@]}; do
    echo ./main_dqn.py $env $algo ${args[@]}

    if [[ "$#" -eq 0 ]]; then
      python -W ignore ./main_dqn.py $env $algo ${args[@]} > /dev/null
    else
      python -W ignore ./main_dqn.py $env $algo ${args[@]}
    fi

    [[ $? -eq 0 ]] && echo "SUCCESS" || echo "FAIL"
  done
done

exit 0
