#!/bin/bash

env=PO-pos-LunarLander-v2
./hpsearch.sh $env --wandb-tag openai

# --negentropy-value-from 1.0 (better than 0.1 and 10.0)
# --optim-lr-actor 0.0003 (better than 0.0001 and 0.001)
# --optim-lr-critic 0.001 (better than 0.0001 and 0.0003)

# some differences in performance, but none solve it

exit 0
