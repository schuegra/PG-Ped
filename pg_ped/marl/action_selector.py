# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:17:40 2019

@author: Philipp
"""

from typing import List, NamedTuple, Callable
import copy
from math import pi as math_pi

import numpy
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn

from pg_ped.marl.utils import split_samples, OrnUhlen as Noise
from pg_ped.environment_construction.state_representation import *
from pg_ped.visualization.visualize_cnn import vis_feature_maps
from pg_ped.environment_construction.geometry import angle_2D_full


# Abstract Action Selector
class ActionSelector(object):

    def __init__(self, policy: nn.Module) -> None:
        self._policy = policy

    def __call__(self, state: Tensor, agent_identity: int, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def copy(self) -> object:
        raise NotImplementedError


# Action Selector Implementations
class ActionSelectorPG(ActionSelector):

    def __init__(self, policy: nn.Module,
                 optimizer: object,
                 episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                 optimize: Callable[[nn.Module, nn.Module, object, List[Tensor]], None],
                 select_action: Callable[[Tensor], Callable[[Tensor], Tensor]],
                 save_model: Callable[[nn.Module, str, int], None]) -> None:
        super().__init__(policy)
        self._optimize = optimize
        self._optimizer = optimizer
        self._episode_discounted_rewards = episode_discounted_rewards
        self._select_action = select_action
        self._save_model = save_model

    def __call__(self, state: List[Tensor], agent_identity: int, **kwargs) -> Tensor:
        return self._select_action(state, self._policy)

    def optimize(self, samples: List[NamedTuple], agent_identity: int, **kwargs) -> None:
        experience_batch = split_samples(samples)

        self._optimize(self._policy,
                       self._episode_discounted_rewards,
                       self._optimizer, *experience_batch,
                       agent_identity, **kwargs)

    def copy(self) -> ActionSelector:
        return copy.deepcopy(self)

    def save_model(self, path: str,
                   agent_identity: int, **kwargs) -> None:
        '''
            Saves the policy parameters to pytorch model.
        '''
        self._save_model(self._policy, path, agent_identity, **kwargs)

    def load_model(self, path: str) -> None:
        '''
            Loads the policy parameters from pytorch model.
        '''
        model = torch.load(path)
        self._policy = model
        # self._policy.load_state_dict(model)


class ActionSelectorPGHeatMaps(ActionSelectorPG):

    def __call__(self, state: List[Tensor],
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 variables_per_agent_per_timestep: int,
                 backward_view: int,
                 rows: int,
                 cols: int,
                 **kwargs) -> Tensor:
        heat_map = render_heat_map(state[0], x_min, x_max, y_min, y_max,
                                   variables_per_agent_per_timestep, backward_view,
                                   rows, cols, **kwargs)
        return self._select_action(heat_map, self._policy)


class ActionSelectorPGGaussianDensities(ActionSelectorPG):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.noise = Noise(2)

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, **kwargs) -> Tensor:
        gaussian_densities = render_density_field(
            state[0], agent_identity,
            x_min, x_max, y_min, y_max,
            variables_per_agent_per_timestep, backward_view,
            rows, cols, **kwargs
        )

        return self._select_action(gaussian_densities, self._policy, forbidden_actions)


class ActionSelectorScenarioMPL(ActionSelectorPG):

    def __init__(self, agent_identity, policy: nn.Module,
                 optimizer: object, episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                 optimize: Callable[[nn.Module, nn.Module, object, List[Tensor]], None],
                 select_action: Callable[[Tensor], Callable[[Tensor], Tensor]],
                 save_model: Callable[[nn.Module, str, int], None],
                 number_agents: int, rows: int, cols: int):
        super().__init__(policy, optimizer, episode_discounted_rewards, optimize,
                         select_action, save_model)

        # To reuse these objects in call, they are stored in this object
        color_current = 0.1
        color_others = 0.2
        my_dpi = 96.
        self._fig, self._ax = plt.subplots(figsize=[cols / my_dpi, rows / my_dpi], dpi=my_dpi)
        self._colors = numpy.array(
            [color_current if i == agent_identity else color_others for i in range(number_agents)]
        )

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float, goal_line: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, **kwargs) -> Tensor:
        scenario = render_vision_field(
            state[0], agent_identity,
            x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view,
            rows, cols, ax=self._ax, fig=self._fig, colors=self._colors, **kwargs
        )
        action_index, log_prob, prob, failed = self._select_action(scenario, self._policy, forbidden_actions)
        return action_index, log_prob, prob, failed


class ActionSelectorScenarioTorchQtd(ActionSelectorPG):

    def __init__(self, policy_net: nn.Module, value_net: nn.Module, optimizer_policy, optimizer_value, *args, **kwargs):
        super().__init__(policy_net, optimizer_policy, *args, **kwargs)
        self._value_net = value_net
        self._optimizer_value = optimizer_value

    def __call__(self, kinematics: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float, goal_line: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, current_episode: int, mode: str,
                 eps_start, eps_end, eps_decay_length, start_learning, **kwargs) -> Tensor:
        scenario = render_vision_field_torch(
            kinematics[0], agent_identity, x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, **kwargs
        )
        action_index, log_prob, prob, failed = self._select_action(scenario, self._policy,
                                                                   forbidden_actions,
                                                                   current_episode, mode, eps_start,
                                                                   eps_end, eps_decay_length, start_learning)
        return action_index, log_prob, prob, failed

    def optimize(self, samples: List[NamedTuple], agent_identity: int, **kwargs) -> None:
        experience_batch = split_samples(samples)

        self._optimize(self._policy, self._value_net, self._episode_discounted_rewards, self._optimizer,
                       self._optimizer_value, *experience_batch, agent_identity, **kwargs)


class ActionSelectorScenarioTorch(ActionSelectorScenarioTorchQtd):

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float, goal_line: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, current_episode: int, mode: str,
                 eps_start, eps_end, eps_decay_length, **kwargs) -> Tensor:
        # number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
        # scenario = render_full_scenario_torch(
        scenario = render_rectangular_fov_torch(
            # scenario = render_vision_field_torch(
            state[0], agent_identity, x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, **kwargs
        )
        speed = torch.norm(state[0][agent_identity, :2] - state[0][agent_identity, 4:6])
        angle = angle_2D_full(state[0][agent_identity, 2:4], torch.tensor([1., 0.], device=state[0].device))
        action_index, log_prob, prob, failed = self._select_action([scenario, speed, angle], self._policy,
                                                                   forbidden_actions,
                                                                   mode, current_episode, eps_start, eps_end,
                                                                   eps_decay_length)
        return action_index, log_prob, prob, failed


class ActionSelectorScenarioTorchContinous(ActionSelectorScenarioTorchQtd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = Noise(2)

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float, goal_line: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, mode: str, eps_decay_length, **kwargs) -> Tensor:
        # if current_step == 0:
        #     self.noise.reset()

        current_episode = kwargs['current_episode']
        # number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step

        # scenario = render_full_scenario_torch(
        # scenario = render_rectangular_fov_torch(
        # scenario = render_vision_field_torch(
        scenario = render_fov_no_rotation(
            state[0], agent_identity, x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, **kwargs
        )
        speed = torch.norm(state[0][agent_identity, :2] - state[0][agent_identity, 4:6])
        angle = angle_2D_full(state[0][agent_identity, 2:4], torch.tensor([1., 0.], device=state[0].device))
        action, log_prob, prob, failed = self._select_action([scenario, speed, angle], self._policy, forbidden_actions,
                                                             mode, current_episode)
        # if kwargs['mode'] == 'train':
        #     action = action + torch.tensor(self.noise.sample(), device=action.device, dtype=torch.float)

        # if agent_identity == 0 and number_of_time_steps_since_training_start < kwargs['start_learning']:
        #     action[0, 1:2] = math_pi/2. * torch.ones(1) + (torch.rand(1) - 0.5)
        return action, log_prob, prob, failed


class ActionSelectorScenarioTorchQtdReevaluate(ActionSelectorScenarioTorchQtd):

    def optimize(self, samples: List[NamedTuple], agent_identity: int, **kwargs) -> None:
        experience_batch = split_samples(samples)

        self._optimize(self, self._episode_discounted_rewards, self._optimizer,
                       self._optimizer_value, *experience_batch, agent_identity, **kwargs)


class ActionSelectorPGGaussDenseLocMap(ActionSelectorPG):

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, window_rows, window_cols, **kwargs) -> Tensor:
        local_gd = render_local_gd(state[0], agent_identity,
                                   x_min, x_max, y_min, y_max,
                                   variables_per_agent_per_timestep,
                                   backward_view, rows, cols,
                                   window_rows, window_cols,
                                   self._ax, self._fig, self._colors, **kwargs)
        action_index, log_prob, prob = self._select_action(local_gd, self._policy, forbidden_actions)
        return action_index, log_prob, prob


class ActionSelectorPGGaussianDensities1StepTD(ActionSelectorPG):

    def __init__(self, policy_net: nn.Module, value_net: nn.Module, optimizer_policy, optimizer_value, *args, **kwargs):
        super().__init__(policy_net, optimizer_policy, *args, **kwargs)
        self._value_net = value_net
        self._optimizer_value = optimizer_value

    def optimize(self, samples: List[NamedTuple], agent_identity: int, **kwargs) -> None:
        experience_batch = split_samples(samples)

        self._optimize(self._policy, self._value_net,
                       self._episode_discounted_rewards,
                       self._optimizer, self._optimizer_value,
                       *experience_batch, agent_identity, **kwargs)

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, **kwargs) -> Tensor:
        gaussian_densities_and_localization_maps = render_gd_and_loc_map(state[0], agent_identity,
                                                                         x_min, x_max, y_min, y_max,
                                                                         variables_per_agent_per_timestep,
                                                                         backward_view,
                                                                         rows, cols, **kwargs)
        action_index, log_prob, prob = self._select_action(gaussian_densities_and_localization_maps, self._policy,
                                                           forbidden_actions)
        state_value = self._value_net(gaussian_densities_and_localization_maps)[0]
        return action_index, log_prob, prob, state_value


class ActionSelectorPGLocalGaussDense1StepTD(ActionSelectorPGGaussianDensities1StepTD):

    def __call__(self, state: List[Tensor], agent_identity: int, forbidden_actions: List[int],
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 variables_per_agent_per_timestep: int, backward_view: int,
                 rows: int, cols: int, window_rows: int, window_cols: int, **kwargs) -> Tensor:
        local_gd = render_local_gd(state[0], agent_identity,
                                   x_min, x_max, y_min, y_max,
                                   variables_per_agent_per_timestep,
                                   backward_view, rows, cols,
                                   window_rows, window_cols, **kwargs)
        action_index, log_prob, prob = self._select_action(local_gd, self._policy, forbidden_actions)
        state_value = self._value_net(local_gd)[0]
        return action_index, log_prob, prob, state_value


class ActionSelectorKinematicsTorchQtd(ActionSelectorPG):

    def __init__(self, policy_net: nn.Module, value_net: nn.Module, optimizer_policy, optimizer_value, *args, **kwargs):
        super().__init__(policy_net, optimizer_policy, *args, **kwargs)
        self._value_net = value_net
        self._optimizer_value = optimizer_value

    def __call__(self, kinematics, forbidden_actions: List[int], current_step: int, mode: str,
                 eps_start, eps_end, eps_decay_length, runner_goal_x, start_learning, episode_lengths,
                 **kwargs) -> Tensor:
        current_overall_step = sum(episode_lengths) + current_step
        action_index, log_prob, prob, failed = self._select_action(kinematics[0], self._policy,
                                                                   forbidden_actions, current_overall_step, mode,
                                                                   eps_start, eps_end, eps_decay_length,
                                                                   start_learning)
        return action_index, log_prob, prob, failed

    def optimize(self, samples: List[NamedTuple], agent_identity: int, **kwargs) -> None:
        experience_batch = split_samples(samples)

        self._optimize(self._policy, self._value_net, self._episode_discounted_rewards, self._optimizer,
                       self._optimizer_value, *experience_batch, agent_identity, **kwargs)


class ActionSelectorKinematicsTorchContinous(ActionSelectorScenarioTorchQtd):

    def __call__(self, kinematics, agent_identity: int, forbidden_actions: List[int], mode: str, **kwargs) -> Tensor:
        current_episode = kwargs['current_episode']
        action, log_prob, prob, failed = self._select_action(kinematics[0], self._policy, forbidden_actions,
                                                             mode, current_episode)
        return action, log_prob, prob, failed


class ActionSelectorKinematicsTorchContinous2(ActionSelectorScenarioTorchQtd):

    def __call__(self, state, agent_identity: int, forbidden_actions: List[int], mode: str, **kwargs) -> Tensor:
        current_episode = kwargs['current_episode']
        #kinematics = generate_kinematics_torch2(state[0], agent_identity, **kwargs)
        kinematics = generate_kinematics_torch(state[0], agent_identity, **kwargs)
        action, log_prob, prob, failed = self._select_action(kinematics[0], self._policy, forbidden_actions,
                                                             mode, current_episode)
        return action, log_prob, prob, failed


class ActionSelectorKinematicsTorchDiscrete(ActionSelectorScenarioTorchQtd):

    def __call__(self, state, agent_identity: int, forbidden_actions: List[int], mode: str, **kwargs) -> Tensor:
        current_episode = kwargs['current_episode']
        #kinematics = generate_kinematics_torch2(state[0], agent_identity, **kwargs)
        kinematics = generate_kinematics_torch(state[0], agent_identity, **kwargs)
        action, log_prob, prob, failed = self._select_action(kinematics, self._policy, forbidden_actions,
                                                             current_episode, mode, None, None, None)
        return action, log_prob, prob, failed