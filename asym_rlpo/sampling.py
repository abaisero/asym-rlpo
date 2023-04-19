import torch

from asym_rlpo.data import Episode, EpisodeBuilder, Interaction
from asym_rlpo.envs import Environment
from asym_rlpo.policies import Policy
from asym_rlpo.utils.convert import numpy2torch


def sample_episode(
    env: Environment,
    policy: Policy,
    *,
    render: bool = False,
) -> Episode:
    with torch.no_grad():
        episode_builder = EpisodeBuilder()

        done = False
        observation, latent = env.reset()
        policy.reset(numpy2torch(observation))

        if render:
            env.render()

        while True:
            action, info = policy.sample_action()
            next_observation, next_latent, reward, done = env.step(action)
            policy.step(torch.tensor(action), numpy2torch(next_observation))

            if render:
                env.render()

            episode_builder.append(
                Interaction(
                    observation=observation,
                    latent=latent,
                    action=action,
                    reward=reward,
                    info=info,
                ),
                done,
            )

            if done:
                break

            latent = next_latent
            observation = next_observation

    return episode_builder.build()


def sample_episodes(
    env: Environment,
    policy: Policy,
    *,
    num_episodes: int,
    render: bool = False,
) -> list[Episode]:
    return [
        sample_episode(env, policy, render=render) for _ in range(num_episodes)
    ]
