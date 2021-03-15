#!/usr/bin/env python

from asym_rlpo.algorithms import make_algorithm
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.stats import standard_error


def main():
    env = make_env('gv_yaml/gv_nine_rooms.13x13.yaml')
    discount = 0.99

    algorithm = make_algorithm('dqn', env)

    num_epochs = 1_000
    num_episodes_training = 2
    num_episodes_evaluation = 20
    num_steps = 100
    evaluation_frequency = 10

    for epoch in range(num_epochs):
        if epoch % evaluation_frequency == 0:
            returns = evaluate(
                env,
                algorithm.target_policy(),
                discount=discount,
                num_episodes=num_episodes_evaluation,
                num_steps=num_steps,
            )
            mean, sem = returns.mean(), standard_error(returns)
            print(f'EVALUATE epoch {epoch} return {mean:.3f} ({sem:.3f})')

        episodes = sample_episodes(
            env,
            algorithm.behavior_policy(),
            num_episodes=num_episodes_training,
            num_steps=num_steps,
        )
        algorithm.process(episodes)


if __name__ == '__main__':
    main()
