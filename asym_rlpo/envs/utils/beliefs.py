from functools import lru_cache

import torch
import gym
from gym_pomdps.envs.pomdp import POMDP

from asym_rlpo.data import Episode


def compute_beliefs(episode: Episode, *, pomdp: gym.Env) -> torch.Tensor:
    belief = compute_init_belief(pomdp)
    beliefs = [belief]
    for a, o in zip(episode.actions[:-1], episode.observations[1:]):
        belief = compute_step_belief(pomdp, belief, a, o)
        beliefs.append(belief)

    return torch.stack(beliefs)


def compute_init_belief(env: gym.Env, shape=None) -> torch.Tensor:
    """Return batch of initial belief-states.

    :param env:  gym.Env environment
    :param shape:  Batch shape of belief-states
    :rtype:  torch.Tensor (*shape, |S|) batch tensor of belief-states
    """

    if not isinstance(env.unwrapped, POMDP):
        raise TypeError('env is not a gym_pomdps.POMDP')

    if shape is None:
        shape = ()

    belief = torch.from_numpy(env.unwrapped.start).float()
    return belief.tile(shape + (1,))


def compute_step_belief(
    env: gym.Env,
    b: torch.Tensor,
    a: torch.Tensor,
    o: torch.Tensor,
) -> torch.Tensor:
    """Return batch of updated belief-states.

    :param env:  gym.Env environment
    :param b:  (*, |S|) batch tensor of belief-states
    :param a:  (*,) batch tensor of actions
    :param o:  (*,) batch tensor of observations
    :rtype:  torch.Tensor (*, |S|) batch tensor of next belief-states
    """

    if not isinstance(env.unwrapped, POMDP):
        raise TypeError('env is not a gym_pomdps.POMDP')

    batch_shape = b.shape[:-1]
    num_states = b.shape[-1]

    if not a.shape == o.shape == batch_shape:
        raise ValueError('Input tensor shapes do not match')

    b = b.reshape(-1, num_states)
    a = a.reshape(-1)
    o = o.reshape(-1)

    if not _plausible(env, b, a, o).all():
        raise ValueError('impossible observation from given belief-action pair')

    p_sa_so = _P_SA_SO(env)[:, a, :, o]
    b1 = torch.einsum('...ij,...i->...j', p_sa_so, b)
    b1 = b1 / b1.sum(-1, keepdim=True)
    b1 = b1.reshape(batch_shape + (num_states,))
    return b1


def _plausible(
    env: gym.Env,
    b: torch.Tensor,
    a: torch.Tensor,
    o: torch.Tensor,
) -> torch.Tensor:
    """Return plausibility of observations following belief-action pair.

    :param env:  gym.Env environment
    :param b:  (*, |S|) batch tensor of belief states
    :param a:  (*,) batch tensor of actions
    :param o:  (*,) batch tensor of observations
    :rtype: torch.Tensor (*,) batch tensor of bools
    """

    if not isinstance(env.unwrapped, POMDP):
        raise TypeError('env is not a gym_pomdps.POMDP')

    p_sa_o = _P_SA_O(env)[:, a, o]
    return torch.einsum('ib,bi->b', p_sa_o, b) > 0.0


@lru_cache(maxsize=None)
def _P_SA_SO(env: gym.Env) -> torch.Tensor:
    """Compute the generative matrix G_{ijkl} = \\Pr(s'=k, o=l \\mid s=i, a=j)

    :param env:  gym.Env environment
    :rtype: torch.Tensor (|S|, |A|, |S|, |O|) batch tensor
    """

    T = torch.from_numpy(env.unwrapped.T).float()
    O = torch.from_numpy(env.unwrapped.O).float()

    temp = O[..., 0].unsqueeze(-1)
    O = torch.concatenate((O, torch.ones_like(temp)), dim=-1)
    return torch.einsum('saz,sazo->sazo', T, O)


@lru_cache(maxsize=None)
def _P_SA_O(env: gym.Env) -> torch.Tensor:
    """Compute the observation matrix O_{ijl} = \\Pr(o=l \\mid s=i, a=j)

    :param env:  gym.Env environment
    :rtype: torch.Tensor (|S|, |A|, |O|) batch tensor of observation probabilities
    """
    return _P_SA_SO(env).sum(-2)
