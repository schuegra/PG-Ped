import itertools
import os

import numpy
import torch

from pg_ped.utils import set_state_vadere


def flatten_list(x, depth=2):
    return [xi for y in x for xi in y]


def start_simulation(episodes, parameter_dict, states, episode_lengths, model_name, use_traci=False):
    print('******************************\n\tSTART SIMULATION\n******************************\n')

    for i, episode in enumerate(episodes):

        if use_traci is True:
            set_state_vadere(episode._single_agent_steps[0]._environment._initial_state)

        parameter_dict['current_episode'] = i
        done = False
        j = 0
        state = episode._single_agent_steps[0]._environment._state
        while done is False:
            states += [state.detach().cpu().numpy()]
            parameter_dict['current_step'] = j
            state, done, _ = episode(**parameter_dict)
            j += 1

        episode_lengths += [j]

    print('******************************\n\tFINISH SIMULATION\n******************************\n')


def evaluate_model(episodes, parameter_dict, states, episode_lengths, model_name, use_traci=False):
    start_simulation(episodes, parameter_dict, states, episode_lengths, model_name, use_traci)
    reward_sums = flatten_list(parameter_dict['reward_sums'][1])
    total_steps = sum(parameter_dict['episode_lengths'])
    return numpy.sum(reward_sums)/total_steps
