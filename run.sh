#!/usr/bin/zsh

for i in $(seq 10); do
  sem -j4 ./main.py fob-dqn "$@"
  sem -j4 ./main.py foe-dqn "$@"
  sem -j4 ./main.py poe-dqn "$@"
  sem -j4 ./main.py poe-adqn "$@"
done

sem --wait

exit 0
