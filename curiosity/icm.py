from abc import ABCMeta, abstractmethod
from typing import Generator

import numpy as np
import torch
from torch import Tensor, nn

from curiosity.base import Curiosity, CuriosityFactory
from envs import Converter
from reporters import Reporter, NoReporter


class ICMModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        super().__init__()
        self.state_converter = state_converter
        self.action_converter = action_converter

    @property
    @abstractmethod
    def recurrent(self) -> bool:
        raise NotImplementedError('Implement me')

    @staticmethod
    @abstractmethod
    def factory() -> 'ICMModelFactory':
        raise NotImplementedError('Implement me')


class ICMModelFactory:
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        raise NotImplementedError('Implement me')


class ForwardModel(nn.Module):
    def __init__(self, action_converter: Converter, state_latent_features: int):
        super().__init__()
        self.action_converter = action_converter
        action_latent_features = 128
        if action_converter.discrete:
            self.action_encoder = nn.Embedding(action_converter.shape[0], action_latent_features)
        else:
            self.action_encoder = nn.Linear(action_converter.shape[0], action_latent_features)
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + state_latent_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_latent_features)
        )

    def forward(self, state_latent: Tensor, action: Tensor):
        action = self.action_encoder(action.long() if self.action_converter.discrete else action)
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x


class InverseModel(nn.Module):
    def __init__(self, action_converter: Converter, state_latent_features: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            action_converter.policy_out_model(128)
        )

    def forward(self, state_latent: Tensor, next_state_latent: Tensor):
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))


class MlpICMModel(ICMModel):
    def __init__(self, state_converter: Converter, action_converter: Converter):
        assert len(state_converter.shape) == 1, 'Only flat spaces supported by MLP model'
        assert len(action_converter.shape) == 1, 'Only flat action spaces supported by MLP model'
        super().__init__(state_converter, action_converter)
        self.encoder = nn.Sequential(
            nn.Linear(state_converter.shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),

        )
        self.forward_model = ForwardModel(action_converter, 128)
        self.inverse_model = InverseModel(action_converter, 128)

    @property
    def recurrent(self) -> bool:
        return False

    def forward(self, state: Tensor, next_state: Tensor, action: Tensor):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    @staticmethod
    def factory() -> 'ICMModelFactory':
        return MlpICMModelFactory()


class MlpICMModelFactory(ICMModelFactory):
    def create(self, state_converter: Converter, action_converter: Converter) -> ICMModel:
        return MlpICMModel(state_converter, action_converter)


class ICM(Curiosity):
    """
    Implements the Intrinsic Curiosity Module described in paper: https://arxiv.org/pdf/1705.05363.pdf

    The overview of the idea is to reward the agent for exploring unseen states. It is achieved by implementing two
    models. One called forward model that given the encoded state and encoded action computes predicts the encoded next
    state. The other one called inverse model that given the encoded state and encoded next_state predicts action that
    must have been taken to move from one state to the other. The final intrinsic reward is the difference between
    encoded next state and encoded next state predicted by the forward module. Inverse model is there to make sure agent
    focuses on the states that he actually can control.
    """

    def __init__(self, state_converter: Converter, action_converter: Converter, model_factory: ICMModelFactory,
                 policy_weight: float, reward_scale: float, weight: float, intrinsic_reward_integration: float,
                 reporter: Reporter):
        """

        :param state_converter: state converter
        :param action_converter: action converter
        :param model_factory: model factory
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to control how
               important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how big the
               intrinsic reward is
        :param weight: balances the importance between forward and inverse model
        :param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward. Used when
               incorporating intrinsic into extrinsic in the ``reward`` method
        :param reporter reporter used to report training statistics
        """
        super().__init__(state_converter, action_converter)
        self.model: ICMModel = model_factory.create(state_converter, action_converter)
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def parameters(self) -> Generator[nn.Parameter, None, None]:
        return self.model.parameters()

    def reward(self, rewards: np.ndarray, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:, :-1], states[:, 1:]
        states, next_states, actions = self._to_tensors(
            self.state_converter.reshape_as_input(states, self.model.recurrent),
            self.state_converter.reshape_as_input(next_states, self.model.recurrent),
            actions.reshape(n * t, *actions.shape[2:]))
        next_states_latent, next_states_hat, _ = self.model(states, next_states, actions)
        intrinsic_reward = self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2)
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy().reshape(n, t)
        self.reporter.scalar('icm/reward',
                             intrinsic_reward.mean().item() if self.reporter.will_report('icm/reward') else 0)
        return (1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_reward

    def loss(self, policy_loss: Tensor, states: Tensor, next_states: Tensor, actions: Tensor) -> Tensor:
        next_states_latent, next_states_hat, actions_hat = self.model(states, next_states, actions)
        forward_loss = 0.5 * (next_states_hat - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()
        inverse_loss = self.action_converter.distance(actions_hat, actions)
        curiosity_loss = self.weight * forward_loss + (1 - self.weight) * inverse_loss
        self.reporter.scalar('icm/loss', curiosity_loss.item())
        return self.policy_weight * policy_loss + curiosity_loss

    def to(self, device: torch.device, dtype: torch.dtype):
        super().to(device, dtype)
        self.model.to(device, dtype)

    @staticmethod
    def factory(model_factory: ICMModelFactory, policy_weight: float, reward_scale: float,
                weight: float, intrinsic_reward_integration: float, reporter: Reporter = NoReporter()) -> 'ICMFactory':
        """
        Creates the factory for the ``ICM``
        :param model_factory: model factory
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to control how
               important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how big the
               intrinsic reward is
        :param weight: balances the importance between forward and inverse model
        :param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward. Used when
               incorporating intrinsic into extrinsic in the ``reward`` method
        :param reporter reporter used to report training statistics
        :return: factory
        """
        return ICMFactory(model_factory, policy_weight, reward_scale, weight, intrinsic_reward_integration, reporter)


class ICMFactory(CuriosityFactory):
    def __init__(self, model_factory: ICMModelFactory, policy_weight: float, reward_scale: float, weight: float,
                 intrinsic_reward_integration: float, reporter: Reporter):
        self.policy_weight = policy_weight
        self.model_factory = model_factory
        self.scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter

    def create(self, state_converter: Converter, action_converter: Converter):
        return ICM(state_converter, action_converter, self.model_factory, self.policy_weight, self.scale, self.weight,
                   self.intrinsic_reward_integration, self.reporter)
