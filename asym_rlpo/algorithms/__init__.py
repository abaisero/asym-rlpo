import torch
import torch.nn as nn

from asym_rlpo.algorithms.a2c import A2C
from asym_rlpo.algorithms.adqn import ADQN, ADQN_VarianceReduced
from asym_rlpo.algorithms.adqn_short import (
    ADQN_Short,
    ADQN_Short_VarianceReduced,
)
from asym_rlpo.algorithms.adqn_state import (
    ADQN_State,
    ADQN_State_VarianceReduced,
)
from asym_rlpo.algorithms.algorithm import ValueBasedAlgorithm
from asym_rlpo.algorithms.biphasic_a2c import BiphasicA2C
from asym_rlpo.algorithms.dqn import DQN
from asym_rlpo.algorithms.mr_a2c import MemoryReactive_A2C
from asym_rlpo.algorithms.noisy_a2c import NoisyA2C
from asym_rlpo.algorithms.trainer import Trainer
from asym_rlpo.data import Episode
from asym_rlpo.models.actor import MemoryReactive_ActorModel
from asym_rlpo.models.actor_critic import (
    MemoryReactive_ActorCriticModel,
    NoisyActorCriticModel,
)
from asym_rlpo.models.critic import H_CriticModel, HM_CriticModel, HZ_CriticModel
from asym_rlpo.models.factory import ModelFactory
from asym_rlpo.models.memory_reactive import (
    MemoryModel,
    MemoryPolicy,
    MemoryReactiveHistoryModel,
)
from asym_rlpo.models.mlp import MLP_Model
from asym_rlpo.models.sequence import make_sequence_model
from asym_rlpo.models.types import CriticType, PolicyModule
from asym_rlpo.modules.mlp import make_mlp
from asym_rlpo.types import OptimizerFactory


def get_a2c_critic_type(name: str) -> CriticType:
    if name == 'a2c':
        return CriticType.H

    if name == 'asym-a2c':
        return CriticType.HZ

    if name == 'asym-a2c-state':
        return CriticType.Z

    raise ValueError(f'invalid algorithm name {name}')


def make_mr_a2c_algorithm(
    name: str,
    model_factory: ModelFactory,
    *,
    actor_optimizer_factory: OptimizerFactory,
    critic_optimizer_factory: OptimizerFactory,
    max_gradient_norm: float,
) -> MemoryReactive_A2C:
    if name != 'mr-a2c':
        raise ValueError(f'invalid mr-a2c {name=}')

    memory_sequence_model_name = 'gru'
    memory_size = 3

    def make_memory_model(memory_size: int) -> MemoryModel:
        interaction_model = model_factory.make_interaction_model()
        sequence_model = make_sequence_model(
            memory_sequence_model_name,
            interaction_model.dim,
            64,
        )
        return MemoryModel(
            interaction_model,
            sequence_model,
            memory_size=memory_size,
        )

    def make_hm_critic_model(
        memory_model: MemoryModel | None = None,
    ) -> HM_CriticModel:
        if memory_model is None:
            memory_model = make_memory_model(memory_size)

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

    memory_model = make_memory_model(memory_size)
    critic_model = make_hm_critic_model(memory_model)
    target_critic_model = make_hm_critic_model()

    memory_policy = MemoryPolicy(critic_model)

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
        model_factory.env.action_space.n,
    )

    actor_model = MemoryReactive_ActorModel(
        memory_reactive_history_model,
        policy_module,
    )

    actor_critic_model = MemoryReactive_ActorCriticModel(
        actor_model,
        critic_model,
    )

    trainer = Trainer.from_factories(
        {
            'actor': actor_optimizer_factory,
            'critic': critic_optimizer_factory,
        },
        {
            'actor': actor_critic_model.actor_model.parameters,
            'critic': actor_critic_model.critic_model.parameters,
        },
        max_gradient_norm=max_gradient_norm,
    )

    return MemoryReactive_A2C(actor_critic_model, target_critic_model, trainer)


def make_a2c_algorithm(
    name: str,
    model_factory: ModelFactory,
    *,
    actor_optimizer_factory: OptimizerFactory,
    critic_optimizer_factory: OptimizerFactory,
    max_gradient_norm: float,
    info: dict,
) -> A2C:
    if name.startswith('biphasic-asym-a2c'):
        return make_biphasic_a2c_algorithm(
            name,
            model_factory,
            actor_optimizer_factory=actor_optimizer_factory,
            critic_optimizer_factory=critic_optimizer_factory,
            max_gradient_norm=max_gradient_norm,
            info=info,
        )

    actor_model = model_factory.make_actor_model()
    critic_type = get_a2c_critic_type(name)
    critic_model = model_factory.make_critic_model(critic_type)
    target_critic_model = model_factory.make_critic_model(critic_type)

    critic_models = nn.ModuleDict({'critic': critic_model})
    target_critic_models = nn.ModuleDict({'critic': target_critic_model})

    trainer = Trainer.from_factories(
        {
            'actor': actor_optimizer_factory,
            'critic': critic_optimizer_factory,
        },
        {
            'actor': actor_model.parameters,
            'critic': critic_model.parameters,
        },
        max_gradient_norm=max_gradient_norm,
    )

    return A2C(actor_model, critic_models, target_critic_models, trainer)


def make_phase_identifier(env: str):
    if 'heavenhell_' in env:
        return phase_identifier_heavenhell

    raise ValueError(f'invalid env {env=} for biphasic-a2c')


def phase_identifier_heavenhell(episode: Episode) -> torch.Tensor:
    dummy_observation = episode.observations[0]
    oracle_observations = [dummy_observation - 2, dummy_observation - 1]
    counts = torch.zeros_like(episode.observations)
    for oracle_observation in oracle_observations:
        counts += episode.observations == oracle_observation

    phases = counts.cumsum(dim=0).sign()
    return phases


def make_biphasic_a2c_algorithm(
    name: str,
    model_factory: ModelFactory,
    *,
    actor_optimizer_factory: OptimizerFactory,
    critic_optimizer_factory: OptimizerFactory,
    max_gradient_norm: float,
    info: dict,
) -> BiphasicA2C:
    if name == 'biphasic-asym-a2c-h-hs':
        critic_type_0 = CriticType.H
        critic_type_1 = CriticType.HZ
    elif name == 'biphasic-asym-a2c-h-s':
        critic_type_0 = CriticType.H
        critic_type_1 = CriticType.Z
    elif name == 'biphasic-asym-a2c-hs-h':
        critic_type_0 = CriticType.HZ
        critic_type_1 = CriticType.H
    elif name == 'biphasic-asym-a2c-s-h':
        critic_type_0 = CriticType.Z
        critic_type_1 = CriticType.H
    else:
        raise ValueError(f'invalid biphasic-a2c {name=}')

    phase_identifier = make_phase_identifier(info['env'])

    actor_model = model_factory.make_actor_model()
    critic_model_0 = model_factory.make_critic_model(critic_type_0)
    critic_model_1 = model_factory.make_critic_model(critic_type_1)
    target_critic_model_0 = model_factory.make_critic_model(critic_type_0)
    target_critic_model_1 = model_factory.make_critic_model(critic_type_1)
    critic_models = nn.ModuleDict(
        {'critic_0': critic_model_0, 'critic_1': critic_model_1}
    )
    target_critic_models = nn.ModuleDict(
        {'critic_0': target_critic_model_0, 'critic_1': target_critic_model_1}
    )

    trainer = Trainer.from_factories(
        {
            'actor': actor_optimizer_factory,
            'critic_0': critic_optimizer_factory,
            'critic_1': critic_optimizer_factory,
        },
        {
            'actor': actor_model.parameters,
            'critic_0': critic_model_0.parameters,
            'critic_1': critic_model_1.parameters,
        },
        max_gradient_norm=max_gradient_norm,
    )

    return BiphasicA2C(
        actor_model,
        critic_models,
        target_critic_models,
        trainer,
        phase_identifier=phase_identifier,
    )


def make_noisy_a2c_algorithm(
    name: str,
    model_factory: ModelFactory,
    *,
    actor_optimizer_factory: OptimizerFactory,
    critic_optimizer_factory: OptimizerFactory,
    max_gradient_norm: float,
) -> NoisyA2C:
    if name != 'noisy-a2c':
        raise ValueError(f'invalid noisy-a2c {name=}')

    h_critic_model = model_factory.make_critic_model(CriticType.H)
    assert isinstance(h_critic_model, H_CriticModel)
    hz_critic_model = model_factory.make_critic_model(CriticType.HZ)
    assert isinstance(hz_critic_model, HZ_CriticModel)

    actor_critic_model = NoisyActorCriticModel(
        model_factory.make_actor_model(),
        h_critic_model,
        hz_critic_model,
    )
    target_h_critic_model = model_factory.make_critic_model(CriticType.H)
    assert isinstance(target_h_critic_model, H_CriticModel)
    target_hz_critic_model = model_factory.make_critic_model(CriticType.HZ)
    assert isinstance(target_hz_critic_model, HZ_CriticModel)

    trainer = Trainer.from_factories(
        {
            'actor': actor_optimizer_factory,
            'h_critic': critic_optimizer_factory,
            'hz_critic': critic_optimizer_factory,
        },
        {
            'actor': actor_critic_model.actor_model.parameters,
            'h_critic': actor_critic_model.h_critic_model.parameters,
            'hz_critic': actor_critic_model.hz_critic_model.parameters,
        },
        max_gradient_norm=max_gradient_norm,
    )

    return NoisyA2C(
        actor_critic_model,
        target_h_critic_model,
        target_hz_critic_model,
        trainer,
        pomdp=pomdp,
    )


def make_dqn_algorithm(
    name: str,
    model_factory: ModelFactory,
    *,
    optimizer_factory: OptimizerFactory,
    max_gradient_norm: float,
) -> ValueBasedAlgorithm:
    if name == 'dqn':
        qha_model = model_factory.make_qha_model()
        target_qha_model = model_factory.make_qha_model()

        trainer = Trainer.from_factories(
            {'qha': optimizer_factory},
            {'qha': qha_model.parameters},
            max_gradient_norm=max_gradient_norm,
        )

        return DQN(
            qha_model=qha_model,
            target_qha_model=target_qha_model,
            trainer=trainer,
        )

    if name == 'adqn':
        qha_model = model_factory.make_qha_model()
        qhza_model = model_factory.make_qhza_model()
        target_qha_model = model_factory.make_qha_model()
        target_qhza_model = model_factory.make_qhza_model()

        trainer = Trainer.from_factories(
            {
                'qha': optimizer_factory,
                'qhza': optimizer_factory,
            },
            {
                'qha': qha_model.parameters,
                'qhza': qhza_model.parameters,
            },
            max_gradient_norm=max_gradient_norm,
        )

        return ADQN(
            qha_model=qha_model,
            qhza_model=qhza_model,
            target_qha_model=target_qha_model,
            target_qhza_model=target_qhza_model,
            trainer=trainer,
        )

    if name == 'adqn-vr':
        qha_model = model_factory.make_qha_model()
        qhza_model = model_factory.make_qhza_model()
        target_qha_model = model_factory.make_qha_model()
        target_qhza_model = model_factory.make_qhza_model()

        trainer = Trainer.from_factories(
            {
                'qha': optimizer_factory,
                'qhza': optimizer_factory,
            },
            {
                'qha': qha_model.parameters,
                'qhza': qhza_model.parameters,
            },
            max_gradient_norm=max_gradient_norm,
        )

        return ADQN_VarianceReduced(
            qha_model=qha_model,
            qhza_model=qhza_model,
            target_qha_model=target_qha_model,
            target_qhza_model=target_qhza_model,
            trainer=trainer,
        )

    if name == 'adqn-state':
        qha_model = model_factory.make_qha_model()
        qza_model = model_factory.make_qza_model()
        target_qha_model = model_factory.make_qha_model()
        target_qza_model = model_factory.make_qza_model()

        trainer = Trainer.from_factories(
            {
                'qha': optimizer_factory,
                'qza': optimizer_factory,
            },
            {
                'qha': qha_model.parameters,
                'qza': qza_model.parameters,
            },
            max_gradient_norm=max_gradient_norm,
        )

        return ADQN_State(
            qha_model=qha_model,
            qza_model=qza_model,
            target_qha_model=target_qha_model,
            target_qza_model=target_qza_model,
            trainer=trainer,
        )

    if name == 'adqn-state-vr':
        qha_model = model_factory.make_qha_model()
        qza_model = model_factory.make_qza_model()
        target_qha_model = model_factory.make_qha_model()
        target_qza_model = model_factory.make_qza_model()

        trainer = Trainer.from_factories(
            {
                'qha': optimizer_factory,
                'qza': optimizer_factory,
            },
            {
                'qha': qha_model.parameters,
                'qza': qza_model.parameters,
            },
            max_gradient_norm=max_gradient_norm,
        )

        return ADQN_State_VarianceReduced(
            qha_model=qha_model,
            qza_model=qza_model,
            target_qha_model=target_qha_model,
            target_qza_model=target_qza_model,
            trainer=trainer,
        )

    if name == 'adqn-short':
        qha_model = model_factory.make_qha_model()
        qhza_model = model_factory.make_qhza_model()
        target_qha_model = model_factory.make_qha_model()
        target_qhza_model = model_factory.make_qhza_model()

        trainer = Trainer.from_factories(
            {
                'qha': optimizer_factory,
                'qhza': optimizer_factory,
            },
            {
                'qha': qha_model.parameters,
                'qhza': qhza_model.parameters,
            },
            max_gradient_norm=max_gradient_norm,
        )

        return ADQN_Short(
            qha_model=qha_model,
            qhza_model=qhza_model,
            target_qha_model=target_qha_model,
            target_qhza_model=target_qhza_model,
            trainer=trainer,
        )

    if name == 'adqn-short-vr':
        qha_model = model_factory.make_qha_model()
        qhza_model = model_factory.make_qhza_model()
        target_qha_model = model_factory.make_qha_model()
        target_qhza_model = model_factory.make_qhza_model()

        trainer = Trainer.from_factories(
            {
                'qha': optimizer_factory,
                'qhza': optimizer_factory,
            },
            {
                'qha': qha_model.parameters,
                'qhza': qhza_model.parameters,
            },
            max_gradient_norm=max_gradient_norm,
        )

        return ADQN_Short_VarianceReduced(
            qha_model=qha_model,
            qhza_model=qhza_model,
            target_qha_model=target_qha_model,
            target_qhza_model=target_qhza_model,
            trainer=trainer,
        )

    raise ValueError(f'invalid algorithm name {name}')
