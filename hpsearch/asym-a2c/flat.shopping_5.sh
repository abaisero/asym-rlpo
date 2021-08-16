#!/bin/bash

env=POMDP-shopping_5-episodic-v1
./hpsearch.sh $env --wandb-tag flat --max-episode-timesteps 100

exit 0
