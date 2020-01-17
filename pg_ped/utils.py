import gc
import math
import os
import pdb
from typing import Tuple, List

import numpy
import torch
import torch.nn as nn
from numpy.random import randint
from pg_ped import config
from pg_ped.helpers import readPersonIDList
from scipy.spatial import distance_matrix as scipy_distance_matrix
from torch import Tensor
from torch.autograd import Variable


def standardize_tensor_nn(tensor: Tensor) -> Tensor:
    out_tensor = torch.zeros_like(tensor)
    batchsize = tensor.shape[0]
    channels = tensor.shape[1]
    for i in range(batchsize):
        for j in range(channels):
            out_tensor[i, j] = (tensor[i, j] - tensor[i, j].mean()) / tensor[i, j].std()
    return out_tensor


def standardize_tensor(tensor: Tensor) -> Tensor:
    out_tensor = (tensor - tensor.mean()) / tensor.std()
    return out_tensor


def normalize_tensor(tensor: Tensor) -> Tensor:
    out_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-9)
    return out_tensor


def prepare(x: List[Tensor]) -> Tensor:
    """
        Converts List of Tensors into a Tensor with shape
        len(List[Tensor]),Tensor.shape[0],Tensor.shape[1],..

        Parameters:
        -----------

        x: List[Tensor]:
            Data
    """
    return torch.cat(x).reshape(len(x), *x[0].shape)


def distance_matrix(positions: Tensor) -> numpy.array:
    """
        Computes the distance for each pair of positions.

        Exploit the symmetry of the distance matrix.

        Parameters:
        -----------

        positions: Tensor
            Has shape (n,2). It represents locations in 2D cartesian coordinates.

    """
    pos = positions.cpu().numpy()
    dists = scipy_distance_matrix(pos, pos)
    return dists


def density(positions: Tensor,
            agent_identity: int,
            influence_radius: float,
            standard_deviation: float) -> float:
    """
        Computes the density based on the distances to other agents.
        The own density is not added.
        The function has a lot of overhead right now, as it computes all densities,
        but only the density at the position of the agent with the given id is
        required.

        Parameters:
        -----------

        positions: Tensor
            Has shape (n,2). It represents locations in 2D cartesian coordinates.
        influence_radius: float
            Positions with greater distance from a point than this radius have
            no effect on the density at a point.

    """

    dists = distance_matrix(positions)[0, :]
    dists = numpy.where(dists == 0, influence_radius + 1, dists)
    densities = numpy.exp(-(dists ** 2) / (2 * standard_deviation))
    densities = numpy.where(dists > influence_radius, 0, densities)
    density = densities.sum()
    return density


def accumulated_density(start: Tensor, goal: Tensor,
                        positions: Tensor,
                        agent_identity: int,
                        influence_radius: float,
                        standard_deviation: float) -> float:
    """

    """

    res = 0.05
    dp = torch.norm(goal - start)
    n_steps = dp / res
    ts = torch.linspace(0, 1, int(n_steps), device=start.device)
    accumulated_density = torch.zeros(1, device=start.device)
    for t in ts:
        positions[agent_identity] = (1 - t) * start + t * goal
        dists = distance_matrix(positions)[0, :]
        dists = numpy.where(dists == 0., influence_radius + 1, dists)
        densities = numpy.exp(-(dists ** 2) / (2 * standard_deviation))
        densities = numpy.where(dists > influence_radius, 0, densities)
        accumulated_density += densities.sum()

    return accumulated_density  # /ts.shape[0]


def set_state_vadere(state):
    ids = readPersonIDList()
    runner_id = ids[-1]
    runner_pos = state[0, :2].cpu().numpy()
    x = runner_pos[0]
    y = runner_pos[1]
    config.cli.pers.setPosition2D(runner_id, x, y)
    for id in ids[:-1]:
        index = int(id)
        pos = state[index, :2].cpu().numpy()
        x = pos[0]
        y = pos[1]
        config.cli.pers.setPosition2D(id, x, y)


def set_random_targets():
    persIDList = config.cli.pers.getIDList()
    polyIDList = config.cli.poly.getIDList()
    targetIDList = []
    for x in polyIDList:
        elementType = config.cli.poly.getType(x)
        if elementType == "TARGET":
            targetIDList += [x]
    randomInts = randint(0, len(targetIDList), 3)
    randomTargetIDs = [targetIDList[x] for x in randomInts]
    for x, y in zip(persIDList, randomTargetIDs):
        if x != persIDList[-1]:
            config.cli.pers.setTargetList(x, y)
            config.cli.pers.setNextTargetListIndex(x, 0)
        else:
            config.cli.pers.setTargetList(x, '6')
            config.cli.pers.setNextTargetListIndex(x, 0)


def get_initial_states_random_on_grid(number_trials: int,
                                      initial_state_runners: List,
                                      x_min: float,
                                      x_max: float,
                                      y_min: float,
                                      y_max: float,
                                      person_radius: float,
                                      number_agents: int,
                                      backward_view: int,
                                      device: str,
                                      **kwargs) -> Tensor:
    initial_states = []
    xs = torch.arange(x_min + person_radius, x_max - person_radius, 2 * person_radius)
    ys = torch.arange(y_min + 4 * person_radius, y_max - person_radius, 2 * person_radius)
    _xs = xs.view(-1, 1).expand(xs.shape[0], ys.shape[0]).reshape(-1)
    _ys = ys.repeat(xs.shape[0])
    positions = torch.stack([_xs, _ys]).transpose(1, 0).numpy()
    for i in range(number_trials):
        sampled = numpy.random.choice(range(positions.shape[0]), number_agents - 1, replace=False)
        angles = 2 * math.pi * numpy.random.rand(number_agents - 1)
        initial_state = [initial_state_runner for initial_state_runner in initial_state_runners] + [
            [positions[sampled[i]][0].copy(), positions[sampled[i]][1].copy(),
             numpy.cos(angles[i]), numpy.sin(angles[i])] * backward_view for i in range(0, number_agents - 1)]
        initial_state = torch.tensor(initial_state, device=device)
        initial_states += [initial_state]
    return initial_states


def get_initial_states_random_on_grid_vadere(number_trials: int,
                                             initial_state_runners: List,
                                             x_min: float,
                                             x_max: float,
                                             y_min: float,
                                             y_max: float,
                                             person_radius: float,
                                             number_agents: int,
                                             backward_view: int,
                                             device: str,
                                             **kwargs) -> Tensor:
    initial_states = []
    xs = torch.arange(x_min + person_radius, x_max - person_radius, 2 * person_radius)
    ys = torch.arange(y_min + person_radius, y_max - person_radius, 2 * person_radius)
    _xs = xs.view(-1, 1).expand(xs.shape[0], ys.shape[0]).reshape(-1)
    _ys = ys.repeat(xs.shape[0])
    positions = torch.stack([_xs, _ys]).transpose(1, 0).numpy()
    for i in range(number_trials):
        sampled = numpy.random.choice(range(positions.shape[0]), number_agents - 1, replace=False)
        angles = 2 * math.pi * numpy.random.rand(number_agents - 1)
        initial_state = [initial_state_runner for initial_state_runner in initial_state_runners] + [
            [positions[sampled[i]][0].copy(), positions[sampled[i]][1].copy(),
             numpy.cos(angles[i]), numpy.sin(angles[i])] * backward_view for i in range(0, number_agents - 1)]
        initial_state = torch.tensor(initial_state, device=device)
        initial_states += [initial_state]
    return initial_states


# def get_initial_states_random_on_grid(number_trials: int,
#                                       initial_state_runners: List,
#                                       x_min: float,
#                                       x_max: float,
#                                       y_min: float,
#                                       y_max: float,
#                                       person_radius: float,
#                                       number_agents: int,
#                                       backward_view: int,
#                                       device: str) -> Tensor:
#     initial_states = []
#     xs = torch.arange(x_min + person_radius, x_max - person_radius, 2 * person_radius)
#     ys = torch.arange(y_min + 4 * person_radius, y_max - person_radius, 2 * person_radius)
#     _xs = xs.view(-1, 1).expand(xs.shape[0], ys.shape[0]).reshape(-1)
#     _ys = ys.repeat(xs.shape[0])
#     positions = torch.stack([_xs, _ys]).transpose(1, 0).numpy()
#     for i in range(number_trials):
#         sampled = numpy.random.choice(range(positions.shape[0]), number_agents - 1, replace=False)
#         initial_state = [initial_state_runner for initial_state_runner in initial_state_runners] + [
#             [positions[sampled[i]][0].copy(), positions[sampled[i]][1].copy(), 0., 0.] *
#             backward_view for i in range(0, number_agents - 1)]
#         initial_state = torch.tensor(initial_state, device=device)
#         initial_states += [initial_state]
#     return initial_states


def get_random_state(x_min: float, x_max: float, y_min: float, y_max: float,
                     person_radius: float, backward_view: int):
    initial_state = []
    for t in range(backward_view):
        sampled_x = numpy.random.uniform(x_min + person_radius, x_max - person_radius)
        sampled_y = numpy.random.uniform(y_min + person_radius, y_max - person_radius)
        initial_state += [sampled_x, sampled_y]
        if t == 0:
            initial_state += [
                0., 0.
            ]
        else:
            initial_state += [
                initial_state[4 * t] - initial_state[4 * (t - 1)],
                initial_state[4 * t + 1] - initial_state[4 * (t - 1) + 1]
            ]

    initial_state_time_corrected = []
    for t in range(backward_view):
        r = backward_view - 1 - t
        initial_state_time_corrected += initial_state[4 * r: 4 * (r + 1)]
    initial_state = [initial_state_time_corrected]

    return initial_state


def get_random_state_fixed_waiting(x_min: float, x_max: float, y_min: float, y_max: float,
                                   person_radius: float, backward_view: int):
    initial_state = []
    sampled_x = numpy.random.uniform(x_min + person_radius, x_max - person_radius)
    sampled_y = numpy.random.uniform(y_min + person_radius, y_max - person_radius)
    for t in range(backward_view):
        initial_state += [sampled_x, sampled_y]
        initial_state += [
            0., 0.
        ]

    return [initial_state]


def get_initial_states_random_with_random_velocity(number_trials: int,
                                                   initial_state_runners: List[List[float]],
                                                   x_min: float,
                                                   x_max: float,
                                                   y_min: float,
                                                   y_max: float,
                                                   person_radius: float,
                                                   number_agents: int,
                                                   backward_view: int,
                                                   device: str) -> Tensor:
    initial_states = []

    for i in range(number_trials):
        initial_state = [initial_state_runner for initial_state_runner in initial_state_runners]
        while len(initial_state) < number_agents:
            initial_state += get_random_state_fixed_waiting(x_min, x_max, y_min, y_max, person_radius, backward_view)

            while check_overlap(initial_state, person_radius):
                initial_state[-1] = \
                    get_random_state_fixed_waiting(x_min, x_max, y_min, y_max, person_radius, backward_view)[0]
        initial_states += [torch.tensor(initial_state, device=device)]

    return initial_states


def check_overlap(state: List[List[float]], person_radius: float):
    for i in range(len(state)):
        for j in range(len(state)):
            if i != j:
                xi, yi = state[i][:2]
                xj, yj = state[j][:2]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if bool(dist < 2 * person_radius):
                    return True
    return False


def get_initial_states_random(number_trials: int,
                              initial_state_runners: List[List[float]],
                              x_min: float,
                              x_max: float,
                              y_min: float,
                              y_max: float,
                              person_radius: float,
                              number_agents: int,
                              backward_view: int,
                              device: str) -> Tensor:
    initial_states = []

    for i in range(number_trials):
        initial_state = [initial_state_runner for initial_state_runner in initial_state_runners]
        while len(initial_state) < number_agents:
            sampled_x = numpy.random.uniform(x_min + person_radius, x_max - person_radius)
            sampled_y = numpy.random.uniform(y_min + person_radius, y_max - person_radius)
            initial_state += [[sampled_x, sampled_y, 0., 0.] * backward_view]
            while check_overlap(initial_state, person_radius):
                sampled_x = numpy.random.uniform(x_min + person_radius, x_max - person_radius)
                sampled_y = numpy.random.uniform(y_min + person_radius, y_max - person_radius)
                initial_state[-1] = [sampled_x, sampled_y, 0., 0.] * backward_view
        initial_states += [torch.tensor(initial_state, device=device)]

    return initial_states


def tensor_contains_nan(tensor: Tensor) -> bool:
    nans = torch.isnan(tensor)
    any = torch.any(nans)
    return bool(any)


def break_if_nan(tensor: Tensor) -> None:
    from .utils import tensor_contains_nan
    nan_present = tensor_contains_nan(tensor)
    if nan_present is True:
        print('Tensor contains nan.')
    return nan_present


def save_pedestrian_model(policy: nn.Module, path: str, agent_identity: int, runner_identities: List[int],
                          **kwargs) -> None:
    '''
        Saves the policy parameters to pytorch model.
    '''
    type_of_pedestrian = "_waiting.pt"
    if agent_identity in runner_identities:
        type_of_pedestrian = "_runner.pt"
    torch.save(policy, path + type_of_pedestrian)


def show_tensors_in_memory():
    # prints currently alive Tensors and Variables
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
