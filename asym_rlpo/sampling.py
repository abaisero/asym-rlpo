from typing import List

import gym
import torch

from asym_rlpo.policies.base import Policy
from asym_rlpo.utils.convert import numpy2torch

from .data import Episode, Interaction, O, S


def sample_episode(
    env: gym.Env,
    policy: Policy,
    *,
    render: bool = False,
) -> Episode[S, O]:
    with torch.no_grad():
        interactions: List[Interaction[S, O]] = []

        done = False
        observation = env.reset()
        state = env.state
        policy.reset(numpy2torch(observation))

        if render:
            env.render()

        while True:
            action = policy.sample_action(numpy2torch(state))
            next_observation, reward, done, _ = env.step(action)
            next_state = env.state
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
    env: gym.Env,
    policy: Policy,
    *,
    num_episodes: int,
    render: bool = False,
) -> List[Episode[S, O]]:
    return [
        sample_episode(env, policy, render=render) for _ in range(num_episodes)
    ]
