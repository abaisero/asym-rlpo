#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

algos=(
  mr-a2c
)

envs=(
  # POMDP-heavenhell_2-episodic-v0
  # POMDP-shopping_5-episodic-v1
  # PO-pos-CartPole-v1
  "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml"
  "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml --gv-ignore-state-channel"
  "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml --gv-ignore-color-channel"
  "gv-yaml/asym-rlpo/gv_four_rooms.7x7.yaml --gv-ignore-state-channel --gv-ignore-color-channel"
  # gv-yaml/asym-rlpo/gv_memory.5x5.yaml
  # gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml
  # "gv-yaml/asym-rlpo/gv_memory.5x5.yaml --latent-type GV-MEMORY"
  # "gv-yaml/asym-rlpo/gv_memory_four_rooms.7x7.yaml --latent-type GV-MEMORY"
  # extra-dectiger-v0
  # extra-cleaner-v0
  # extra-car-flag-v0
)

args=(
  --max-simulation-timesteps 500
  --max-episode-timesteps 100
  --simulation-num-episodes 2

  --history-model rnn
  # --history-model attention
  # --attention-num-heads 1
  # --truncated-histories-n 10

  # --normalize-hs-features
  # --hs-features-dim 64

  --gv-cnn "$PWD/gv-cnns/gv-cnn.v1.yaml"
  --gv-state-submodels agent-grid-cnn agent item
  --gv-state-representation-layers 0
  --gv-observation-submodels grid-cnn item
  --gv-observation-representation-layers 0

  --negentropy-value-from 0.1
  --negentropy-value-to 0.01
  --negentropy-nsteps 1_000_000
)

silent=true

if [[ "$1" == "-v" ]]; then
  shift
  echo "running with standard output"
  echo
  silent=false
else
  echo "running without standard output"
  echo
  silent=true
fi

if [[ "$1" == "--debug" ]]; then
  shift
  echo "running with debugging"
  echo
  debug="-m ipdb -c continue"
  silent=false
fi

if [ "$silent" = true ]; then
  outstream="/dev/null";
else
  outstream="/dev/stdout";
fi

warnings="-W ignore"

for env in "${envs[@]}"; do
  for algo in "${algos[@]}"; do
    command="python $warnings $debug ./main_mr_a2c.py $env $algo ${args[*]} $*"
    echo "$command"
    
    if $command >$outstream; then
      echo "SUCCESS"
    else
      echo "FAIL"
    fi

  done
done

exit 0
