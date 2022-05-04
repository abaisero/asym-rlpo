from typing import List

import torch

from asym_rlpo.envs import Environment
from asym_rlpo.policies.base import Policy
from asym_rlpo.utils.convert import numpy2torch

from .data import Episode, Interaction


def sample_episode(
    env: Environment,
    policy: Policy,
    *,
    render: bool = False,
) -> Episode:
    with torch.no_grad():
        interactions: List[Interaction] = []

        done = False
        state, observation = env.reset()
        policy.reset(numpy2torch(observation))

        if render:
            env.render()

        while True:
            action = policy.sample_action(numpy2torch(state))
            next_state, next_observation, reward, done = env.step(action)
            policy.step(torch.tensor(action), numpy2torch(next_observation))

            if render:
                env.render()

            interactions.append(
                Interaction(
                    state=state,
                    observation=observation,
                    action=action,
                    reward=reward,
                )
            )

            if done:
                break

            state = next_state
            observation = next_observation

    return Episode.from_interactions(interactions)


def sample_episodes(
    env: Environment,
    policy: Policy,
    *,
    num_episodes: int,
    render: bool = False,
) -> List[Episode]:
    return [
        sample_episode(env, policy, render=render) for _ in range(num_episodes)
    ]
