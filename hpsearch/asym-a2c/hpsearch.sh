#!/bin/bash

env=$1
shift
env_label=$(basename $env)
nseeds=4
ntasks=$(($nseeds * 3 * 3 * 3))

# non-truncated histories
algos=(a2c asym-a2c asym-a2c-state)
for algo in ${algos[@]}; do
  algo_label=$algo
  job_name=$env_label.$algo_label
  sbatch --ntasks $ntasks --job-name $job_name \
    hpsearch.sbatch $env $env_label $algo $algo_label $nseeds \
    "$@"
done

# truncated histories
algo=a2c
ns=(2 4)
for n in ${ns[@]}; do
  algo_label=a2c-reactive-$n
  job_name=$env_label.$algo_label
  sbatch --ntasks $ntasks --job-name $job_name \
    hpsearch.sbatch $env $env_label $algo $algo_label $nseeds \
    --truncated-histories --truncated-histories-n $n \
    "$@"
done

exit 0
