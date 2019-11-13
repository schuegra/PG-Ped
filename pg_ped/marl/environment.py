# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:08:08 2019

@author: Philipp
"""

from typing import List, Callable, Tuple, Dict

import torch
from torch import Tensor


class Environment(object):

    @property
    def state(self):
        return self._state

    def done(self, **kwargs) -> bool:
        for crit in self._termination_criteria:
            if crit(self.state, **kwargs) is True:
                self._done = True
                print('Environment done: ', crit.__name__)
                break
        return self._done

    def __init__(self, initial_state: Tensor,
                 state_transition: Callable[[Tensor, int, Tensor], Tensor],
                 reward_function: Callable[[Tensor, int, object], Tensor],
                 termination_criteria: List[Callable[[Tensor], bool]]) -> None:
        self._state = initial_state
        self._initial_state = initial_state.clone()
        self._state_transition = state_transition
        self._reward_function = reward_function
        self._termination_criteria = termination_criteria
        self._done = False

    def step(self, action: Tensor, agent_identity: int, **kwargs: object) -> Tuple[Tensor, Tensor]:
        kwargs['initial_state'] = self._initial_state
        self._state, failed = self._state_transition(self._state, agent_identity, action, **kwargs)
        reward = self._reward_function(self._state, agent_identity, **kwargs)
        return self._state, reward, failed

    def reset(self, initial_state: Tensor):
        self._initial_state = initial_state
        self._done = False

    def __str__(self) -> str:
        string = ""
        string += "Environment terminated: " + str(self.done) + "\n"
        string += "State: " + str(self._state)
        return string
