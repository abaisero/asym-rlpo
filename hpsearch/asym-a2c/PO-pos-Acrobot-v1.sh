#!/bin/bash

env=PO-pos-Acrobot-v1
./hpsearch.sh $env --wandb-tag openai --max-simulation-timesteps 1_000_000

# --negentropy-value-from 0.1 (better than 1.0 and 10.0)
# --optim-lr-actor 0.0003 (better than 0.0001 and 0.001)
# --optim-lr-critic 0.0003 (better than 0.0001 and 0.001)

# major differences in performance

exit 0
