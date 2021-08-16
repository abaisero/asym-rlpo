#!/bin/bash

env=POMDP-heavenhell_4-episodic-v0
./hpsearch.sh $env --wandb-tag flat --max-episode-timesteps 100 --max-simulation-timesteps 10_000_000

exit 0
