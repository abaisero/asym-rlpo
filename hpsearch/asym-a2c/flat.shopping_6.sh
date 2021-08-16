#!/bin/bash

env=POMDP-shopping_6-episodic-v1
./hpsearch.sh $env --wandb-tag flat --max-episode-timesteps 100

exit 0
