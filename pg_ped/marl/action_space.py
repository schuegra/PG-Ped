# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:50:04 2019

@author: Philipp
"""

from typing import Tuple, List, Callable

import itertools

import matplotlib.pyplot as plt
from matplotlib import rc;

import numpy

import torch
from torch import Tensor


# Basic Action Space     
class ActionSpace(object):

    @property
    def shape(self) -> Tuple[int]:
        return self._actions.shape

    def __init__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        if type(self._actions) is list:
            length = len(self._actions)
        else:
            length = self._actions.shape[0]
        return length

    def __call__(self, index: int) -> Tensor:
        try:
            return self._actions[index]
        except Exception as e:
            print('CUDA ERROR')

    def __str__(self) -> str:
        raise NotImplementedError


# Domain Specific Action Spaces, overriding Action Space's init.
class ActionSpace1D(ActionSpace):

    def __init__(self,
                 number_actions: int) -> None:
        self._number_actions = number_actions
        self._actions = torch.tensor([i for i in range(number_actions)])

    def __str__(self) -> str:
        string = "Number actions: " + str(self._number_actions)
        return string


class ActionSpaceDiscreteSteps(ActionSpace):

    def __init__(self,
                 angles: List[float],
                 speeds: List[float],
                 device: str,
                 augment_null: bool = True) -> None:
        self._angles = angles
        self._speeds = speeds

        speed_axis = torch.tensor(speeds).to(device)
        rotation_axis = torch.tensor(angles).to(device)

        # Array of all possible combinations of speed and rotation
        _speed_axis = speed_axis.view(-1, 1). \
            expand(speed_axis.shape[0], rotation_axis.shape[0]). \
            reshape(-1)
        _rotation_axis = rotation_axis.repeat(speed_axis.shape[0])
        movements = torch.stack([_speed_axis, _rotation_axis]).transpose(1, 0)
        stand_still = torch.zeros(1, 2).to(device)
        if augment_null is True:
            self._actions = torch.cat([stand_still, movements])
        else:
            self._actions = movements

    def __str__(self) -> str:
        string = "Angles: " + str(self._angles)
        string += "\nSpeeds: " + str(self._speeds)
        string += "\nStand still: this action space includes the action of not moving."
        return string


class ActionSpaceDiscretePushingSteps(ActionSpace):

    def __init__(self,
                 angles: List[float],
                 speeds: List[float],
                 push_dists: List[float],
                 push_funcs: Callable[[Tensor], Tensor],
                 device: str) -> None:
        self._angles = angles
        self._speeds = speeds
        self._push_dists = push_dists
        self._push_funcs = push_funcs

        def do_nothing(*args, **kwargs):
            pass

        actions = [(0., 0., 0., do_nothing)]
        for action in itertools.product(angles, speeds, push_dists, push_funcs):
            actions += [action]

        self._actions = actions

    def __str__(self) -> str:
        string = "Angles: " + str(self._angles)
        string += "\nSpeeds: " + str(self._speeds)
        string += "\nStand still: this action space includes the action of not moving."
        return string


class ActionSpace2D(ActionSpace):

    def __init__(self,
                 max_speed: float,
                 max_rotation: float,
                 least_speed_diff: float,
                 least_rotation_diff: float) -> None:
        self._max_speed = max_speed
        self._max_rotation = max_rotation
        self._least_speed_diff = least_speed_diff
        self._least_rotation_diff = least_rotation_diff

        speed_axis = torch.arange(-max_speed,
                                  max_speed + least_speed_diff,
                                  least_speed_diff)
        rotation_axis = torch.arange(-max_rotation,
                                     max_rotation + least_rotation_diff,
                                     least_rotation_diff)

        # Array of all possible combinations of speed and rotation
        _speed_axis = speed_axis.view(-1, 1). \
            expand(speed_axis.shape[0], rotation_axis.shape[0]). \
            reshape(-1)
        _rotation_axis = rotation_axis.repeat(speed_axis.shape[0])
        actions = torch.stack([_speed_axis, _rotation_axis]).transpose(1, 0)
        actions_corrected = torch.cat([actions[actions[:, 0] != 0], torch.zeros(1, 2)])
        self._actions = actions_corrected

    def __str__(self) -> str:
        string = ""
        string += "Max speed (m/s): " + str(round(self._max_speed, 5)) + "\n"
        string += "Max rotation (rad): " + str(round(self._max_rotation, 5)) + "\n"
        string += "Least difference in speed (m/s): " + str(round(self._least_speed_diff, 5)) + "\n"
        string += "Least difference in rotation (rad): " + str(round(self._least_rotation_diff, 5)) + "\n"
        return string
