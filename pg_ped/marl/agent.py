# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:02:19 2019

@author: Philipp
"""

from typing import List, NamedTuple, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from .action_space import ActionSpace
from .action_selector import ActionSelector
from .utils import Transition

DUMMY_AGENT_INDEX = torch.tensor(-1)

class ReplayMemory(object):

    def __init__(self, sampling_method: Callable[[List[NamedTuple], object, int], List[NamedTuple]],
                 capacity: int) -> None:
        self._sampling_method = sampling_method
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def push(self, state: Tensor,
             action: Tensor,
             action_index: Tensor,
             log_prob: Tensor,
             prob: Tensor,
             next_state: Tensor,
             reward: Tensor,
             state_value: Tensor = None,
             next_state_value: Tensor = None) -> None:
        """Saves a transition."""

        if self._capacity == 0:
            del state, action, action_index, log_prob, prob, next_state, reward, state_value, next_state_value
            return

        if len(self._memory) < self._capacity:
            self._memory.append(None)

        to_save = [state, action, action_index, log_prob, prob, next_state, reward, state_value, next_state_value]
        self._memory[self._position] = Transition(*to_save)
        self._position = (self._position + 1) % self._capacity

    def sample(self, time_steps: int, num_samples: int) -> List[NamedTuple]:
        return self._sampling_method(self._memory, time_steps, num_samples, self._position)

    def __len__(self) -> int:
        return len(self._memory)


class Agent(object):

    def __init__(self, identity: int, type: int,
                 action_space: ActionSpace,
                 action_selector: ActionSelector,
                 sampling_method: Callable[[List[NamedTuple], object], List[NamedTuple]],
                 capacity: int = 300,
                 trainable: bool = True) -> None:
        self.identity = identity
        self.type = type
        self._replay_memory = ReplayMemory(sampling_method, capacity)
        self._action_space = action_space
        self._action_selector = action_selector
        self._trainable = trainable

    def action(self, state: Tensor, forbidden_actions: List[int], **kwargs) -> Tensor:
        action_index, log_prob, prob, failed = self._action_selector((state,), self.identity, forbidden_actions, **kwargs)
        if failed is False:
            action = self._action_space(action_index)
        else:
            action = -1
        return action, action_index, log_prob, prob, failed

    def optimize(self, sample_size: int, **kwargs) -> None:
        if self._trainable is True:
            samples = self._replay_memory.sample(sample_size, sample_size)
            kwargs['number_samples'] = sample_size
            self._action_selector.optimize(samples, self.identity, **kwargs)

    def observe(self, state: Tensor,
                action: Tensor,
                action_index: Tensor,
                log_prob: Tensor,
                prob: Tensor,
                next_state: Tensor,
                reward: Tensor,
                *args) -> None:
        self._replay_memory.push(state,
                                 action,
                                 action_index,
                                 log_prob,
                                 prob,
                                 next_state,
                                 reward)

    def save_model(self, path: str, **kwargs) -> None:
        '''
            Saves the policy parameters to pytorch model.
        '''
        self._action_selector.save_model(path, self.identity, **kwargs)

    def load_model(self, path: str) -> None:
        '''
            Loads the policy parameters from pytorch model.
        '''
        self._action_selector.load_model(path)

    def __str__(self):
        return "ID: " + str(self.identity)


class Agent1Step(Agent):

    def observe(self, state: Tensor,
                action: Tensor,
                action_index: Tensor,
                log_prob: Tensor,
                prob: Tensor,
                next_state: Tensor,
                reward: Tensor,
                state_value: Tensor,
                next_state_value: Tensor) -> None:
        self._replay_memory.push(state, action, action_index, log_prob, prob, next_state, reward,
                                 state_value, next_state_value)


class AgentMixedSampling(Agent1Step):

    def optimize(self, time_steps, number_samples, **kwargs) -> None:
        if self._trainable is True:
            samples = self._replay_memory.sample(time_steps, number_samples)
            kwargs['number_samples'] = number_samples
            self._action_selector.optimize(samples, self.identity, **kwargs)


class AgentContinousAction(AgentMixedSampling):

    def action(self, state: Tensor, forbidden_actions: List[int], **kwargs) -> Tensor:
        action, log_prob, prob, failed = self._action_selector((state,), self.identity, forbidden_actions, **kwargs)
        return action, DUMMY_AGENT_INDEX.to(action.device), log_prob, prob, failed


class AgentMixedAction(Agent):

    def action(self, state: Tensor, **kwargs) -> Tensor:
        action, log_prob = self._action_selector((state,), self.identity, **kwargs)
        return action, torch.zeros(1), log_prob


class AgentActorCritic(Agent):

    def action(self, state: Tensor, forbidden_actions: List[int], **kwargs) -> Tensor:
        action_index, log_prob, prob, action_value = self._action_selector((state,), self.identity,
                                                                           forbidden_actions, **kwargs)
        action = self._action_space(action_index)
        return action, action_index, log_prob, prob, action_value

    def optimize(self, num_samples, **kwargs) -> None:
        samples = self._replay_memory.sample(num_samples)
        self._action_selector.optimize(samples, self.identity, **kwargs)

    def observe(self, state: Tensor,
                action: Tensor,
                action_index: Tensor,
                log_prob: Tensor,
                prob: Tensor,
                next_state: Tensor,
                reward: Tensor,
                state_value: Tensor,
                next_state_value: Tensor) -> None:
        self._replay_memory.push(state, action, action_index, log_prob,
                                 next_state, reward, state_value, next_state_value, prob)
