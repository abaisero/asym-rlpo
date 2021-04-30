from typing import Dict, List

import gym
import numpy as np
import torch

from asym_rlpo.policies.base import Policy
from asym_rlpo.utils.convert import numpy2torch

from .data import Episode, Interaction, RawEpisode

# gridverse types
GV_State = Dict[str, np.ndarray]
GV_Observation = Dict[str, np.ndarray]
GV_Interaction = Interaction[GV_State, GV_Observation]
GV_Episode = Episode[GV_State, GV_Observation]


def sample_episode(
    env: gym.Env,
    policy: Policy,
    *,
    render: bool = False,
) -> GV_Episode:
    with torch.no_grad():
        interactions: List[GV_Interaction] = []

        start, done = True, False
        observation = env.reset()
        state = env.state
        policy.reset(numpy2torch(observation))

        if render:
            env.render()

        while not done:
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
                    start=start,
                    done=done,
                )
            )

            state = next_state
            observation = next_observation
            start = False

    return Episode.from_raw_episode(RawEpisode(interactions))


def sample_episodes(
    env: gym.Env,
    policy: Policy,
    *,
    num_episodes: int,
    render: bool = False,
) -> List[GV_Episode]:
    return [
        sample_episode(env, policy, render=render) for _ in range(num_episodes)
    ]
