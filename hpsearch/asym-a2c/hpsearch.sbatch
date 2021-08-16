#!/bin/bash

#SBATCH --output outputs/slurm.%x.%j.out
#SBATCH --partition short
#SBATCH --time 24:00:00

#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --mail-user baisero.a@husky.neu.edu

export PYTHONPATH="~/git/asym-rlpo/:$PYTHONPATH"

env=$1
shift
env_label=$1
shift
algo=$1
shift
algo_label=$1
shift
nseeds=$1
shift

script="$HOME/git/asym-rlpo/main_a2c.py"
args=(
  $env
  $algo
  --env-label $env_label
  --algo-label $algo_label
  --simulation-num-episodes 2
  --negentropy-value-to-factor 0.1
  --q-estimator td-lambda
  --q-estimator-lambda 0.7
  --wandb-offline
  --wandb-tag "hpsearch"
  --wandb-tag "discovery"
  --wandb-tag "user=$USER"
)
params_optim_lr_actor=(0.0001 0.0003 0.001)
params_optim_lr_critic=(0.0001 0.0003 0.001)
params_negentropy_value_from=(0.1 0.3 1.0)

for seed in $(seq $nseeds); do
  i=0
  for optim_lr_actor in ${params_optim_lr_actor[@]}; do
    for optim_lr_critic in ${params_optim_lr_critic[@]}; do
      for negentropy_value_from in ${params_negentropy_value_from[@]}; do
        wandb_group="hp=$i;lr-a=$optim_lr_actor;lr-c=$optim_lr_critic;neg-vf=$negentropy_value_from"
        i=$(($i+1))

        srun --ntasks 1 --nodes 1 --exclusive \
          $script ${args[@]} \
          --seed $seed \
          --optim-lr-actor $optim_lr_actor \
          --optim-lr-critic $optim_lr_critic \
          --negentropy-value-from $negentropy_value_from \
          --wandb-group $wandb_group \
          "$@" &
      done
    done
  done
done

wait

exit 0
