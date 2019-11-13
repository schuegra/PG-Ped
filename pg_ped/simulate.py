import os
import itertools

import numpy

import torch


def flatten_list(x, depth=2):
    return [xi for y in x for xi in y]

def start_simulation(episodes, parameter_dict, states, episode_lengths, model_name):

    print('******************************\n\tSTART SIMULATION\n******************************\n')

    for i, episode in enumerate(episodes):
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


def evaluate_model(episodes, parameter_dict, states, episode_lengths, model_name):

    start_simulation(episodes, parameter_dict, states, episode_lengths, model_name)
    reward_sums = flatten_list(parameter_dict['reward_sums'])
    return numpy.mean(reward_sums)