from typing import List, Optional, Tuple
from copy import deepcopy
import itertools
from functools import partial

import torch
from torch.multiprocessing import Pool, Process, set_start_method

from pg_ped import train, simulate


def split_episodes(episodes: List[torch.Tensor], train_ratio: Optional[float] = 0.9) -> Tuple[torch.Tensor]:
    '''Split episodes into train and test data.

    '''

    split = int(train_ratio * len(episodes))
    return episodes[:split], episodes[split:]


def prepare_evaluation(parameter_dict, episodes):
    parameter_dict_deepcopy = deepcopy(parameter_dict)
    episodes_deepcopy = deepcopy(episodes)
    episodes_train, episodes_test = split_episodes(episodes_deepcopy)
    for et in episodes_test:
        et._training = False
    return parameter_dict_deepcopy, episodes_train, episodes_test


def record_results(results, mean_reward, v):
    result = {'mean_reward': mean_reward, 'discount_factor': v[0], 'initial_learning_rate_runner': v[1],
              'initial_learning_rate_waiting': v[1], 'n_hiddens': v[2], 'n_neurons': v[3], 'activation': str(v[4]),
              'dropout_probability': v[4]}
    results += [result]


def evaluate_hyperparameters(vi, results, episodes, parameter_dict, losses, reward_sums, episode_lengths,
                             number_episodes, number_agents, model_name, model_path, plot=True):
    print('EVALUATE HYPERPARAMETERS ', vi)
    parameter_dict['discount_factor'] = vi[0]
    parameter_dict['learning_rate_runner_policy'] = vi[1]
    parameter_dict['learning_rate_waiting_policy'] = vi[1]

    for sas in episodes[0]._single_agent_steps:
        sas._agent._action_selector._value_net.reconstruct(*vi[2:])
        sas._agent._action_selector._policy.reconstruct(*vi[2:])

    parameter_dict_deepcopy, episodes_train, episodes_test = prepare_evaluation(parameter_dict, episodes)
    parameter_dict_deepcopy['mode'] = 'train'
    train.start_training(episodes_train, parameter_dict_deepcopy, losses, reward_sums, episode_lengths,
                         number_episodes, number_agents, model_name, model_path,
                         optimizer_message_frequency=number_episodes + 1, plot=False)

    parameter_dict_deepcopy['mode'] = 'simulate'
    reward_sums = [[] for a in range(number_agents)]
    parameter_dict_deepcopy['reward_sums'] = reward_sums
    mean_reward = simulate.evaluate_model(episodes_test, parameter_dict_deepcopy, [], episode_lengths, model_name)
    print(mean_reward)
    record_results(results, mean_reward, vi)


def start_grid_search(episodes, parameter_dict, losses, reward_sums, episode_lengths,
                      number_episodes, number_agents, model_name, model_path, plot=True):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    grid = dict()
    grid['discount_factors'] = [0.9, 0.95, 0.99]
    grid['initial_learning_rates'] = [1e-3, 1e-4, 1e-5]
    grid['n_hiddens'] = [1, 2, 3]
    grid['n_neurons'] = [[64], [64, 64], [64, 128, 64]]
    grid['activations'] = [torch.nn.ReLU(), torch.nn.Tanh()]
    grid['dropout_probabilities'] = [0.1, 0.2, 0.3]

    results = []
    kwargs = {
        'results': results,
        'episodes': episodes,
        'parameter_dict': parameter_dict,
        'losses': losses,
        'reward_sums': reward_sums,
        'episode_lengths': episode_lengths,
        'number_episodes': number_episodes,
        'number_agents': number_agents,
        'model_name': model_name,
        'model_path': model_path
    }
    hyperparameters = []
    for v0 in grid['discount_factors']:
        for v1 in grid['initial_learning_rates']:
            for i, v2 in enumerate(grid['n_hiddens']): # and v3
                for v4 in grid['activations']:
                    for v5 in grid['dropout_probabilities']:
                        hyperparameters += [[v0, v1, v2, grid['n_neurons'][i], v4, v5]]


    for hp in hyperparameters:
        evaluate_hyperparameters(hp, **kwargs)
    # pool = Pool(1)
    # out1, out2, out3 = zip(*pool.starmap(partial(evaluate_hyperparameters, **kwargs), hyperparameters))

    i_best_result = -1
    best_result = -1e10
    for i, result in enumerate(results):
        if result['mean_reward'] > best_result:
            best_result = result['mean_reward']
            i_best_result = i

    return results[i_best_result]