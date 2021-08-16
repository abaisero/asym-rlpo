#!/bin/bash

env=POMDP-rock_sample_5_6-episodic-v2
./hpsearch.sh $env --wandb-tag flat --max-episode-timesteps 100

exit 0
