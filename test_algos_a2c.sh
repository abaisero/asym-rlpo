#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

envs=(
  # POMDP-heavenhell-episodic-v0
  # POMDP-shopping_5-episodic-v1
  # PO-pos-CartPole-v1
  gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml
  gv-yaml/asym-rlpo/gv_memory.5x5.yaml
  gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml
  "gv-yaml/asym-rlpo/gv_memory.5x5.yaml --latent-type GV-MEMORY"
  "gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml --latent-type GV-MEMORY"
  # extra-dectiger-v0
  # extra-cleaner-v0
  # extra-car-flag-v0
)

algos=(
  a2c
  asym-a2c
  # asym-a2c-state
)

args=(
  --max-simulation-timesteps 500
  --max-episode-timesteps 100
  --simulation-num-episodes 2
  # --truncated-histories
  # --truncated-histories-n 10
  # --normalize-hs-features
  # --hs-features-dim 64
  # --gv-state-model-type cnn
  --gv-state-grid-model-type fc
  --gv-state-representation-layers 2
  --gv-observation-grid-model-type fc
  --gv-observation-representation-layers 2
)

WARNINGS="-W ignore"
# WARNINGS=""

if [[ "$1" == "-v" ]]; then
  shift
  echo "running with standard output"
  echo
  SILENT=false
else
  echo "running without standard output"
  echo
  SILENT=true
fi

DEBUG=""

if [[ "$1" == "--debug" ]]; then
  shift
  echo "running with debugging"
  echo
  DEBUG="-m ipdb -c continue"
  SILENT=false
fi

if $SILENT; then 
  CMD_REDIRECT=/dev/null
else 
  CMD_REDIRECT=/dev/stdout
fi

for env in "${envs[@]}"; do
  for algo in "${algos[@]}"; do
    cmd="python $WARNINGS $DEBUG ./main_a2c.py $env $algo ${args[*]} $*"
    echo "$cmd"

    if $cmd > $CMD_REDIRECT; then
      echo "SUCCESS"
    else
      echo "FAIL"
    fi
  done
done

exit 0
