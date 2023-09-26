import pytest
import torch

from asym_rlpo.envs import make_env
from asym_rlpo.models import make_model_factory
from asym_rlpo.models.actor import ActorModel
from asym_rlpo.models.actor_critic import ActorCriticModel
from asym_rlpo.models.memory_reactive import (
    HM_CriticModel,
    MemoryModel,
    MemoryPolicy,
    MemoryReactiveHistoryModel,
)
from asym_rlpo.models.mlp import MLP_Model
from asym_rlpo.models.sequence import make_sequence_model
from asym_rlpo.models.types import PolicyModule
from asym_rlpo.modules.mlp import make_mlp
from asym_rlpo.sampling import sample_episode


@pytest.mark.parametrize('history_model_name', ['rnn'])
@pytest.mark.parametrize('memory_size', [3])
def test_memory_model(history_model_name: str, memory_size: int):
    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type='state',
        max_episode_timesteps=100,
    )

    model_factory = make_model_factory(env)
    model_factory.history_model = history_model_name
    model_factory.attention_num_heads = 1
    model_factory.history_model_memory_size = 0

    def make_memory_model(memory_size: int) -> MemoryModel:
        interaction_model = model_factory.make_interaction_model()
        sequence_model = make_sequence_model(
            history_model_name,
            interaction_model.dim,
            64,
        )
        return MemoryModel(
            interaction_model,
            sequence_model,
            memory_size=memory_size,
        )

    memory_model = make_memory_model(memory_size)

    def make_hm_critic_model() -> HM_CriticModel:
        history_model = model_factory.make_history_model()
        value_module = MLP_Model(
            [history_model.dim + memory_model.dim, 256, 1],
            ['relu', 'identity'],
        )
        return HM_CriticModel(
            history_model,
            memory_model,
            value_module,
        )

    critic_model = make_hm_critic_model()

    memory_policy = MemoryPolicy(critic_model)
    memory_policy.epsilon = 0.5

    memory_reactive_history_model = MemoryReactiveHistoryModel(
        memory_model,
        memory_policy,
    )

    def make_memory_reactive_policy_module(
        in_size: int,
        out_size: int,
    ) -> PolicyModule:
        return make_mlp([in_size, 512, out_size], ['relu', 'logsoftmax'])

    policy_module = make_memory_reactive_policy_module(
        memory_reactive_history_model.dim,
        env.action_space.n,
    )

    actor_model = ActorModel(
        memory_reactive_history_model,
        policy_module,
    )

    policy = actor_model.policy()
    episode = sample_episode(env, policy).torch()

    memories = episode.info['memory']
    assert (memories <= torch.arange(memories.numel())).all()

    action_logits = actor_model.action_logits(episode)
    assert action_logits.size(0) == len(episode)

    values = critic_model.values(episode)
    assert values.numel() == len(episode)

    actor_critic_model = ActorCriticModel(
        actor_model,
        critic_model,
    )

    # print(values)
    # print(action_logits)
