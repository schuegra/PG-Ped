import os
import time
from copy import deepcopy
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy
import yaml
from pg_ped import traci_store, config
from pg_ped.evaluation.evaluate_trajectories import extract_trajectories, save_trajectories
from pg_ped.visualization.plot_trajectories import plot_trajectories


def setup_experiment_directory(model_name, model_path, curves_path):
    if not os.path.exists(model_name):
        os.mkdir(model_name)
        if not os.path.exists(os.path.join(model_name, curves_path)):
            os.mkdir(os.path.join(model_name, curves_path))
        if not os.path.exists(os.path.join(model_name, model_path)):
            os.mkdir(os.path.join(model_name, model_path))


def create_fn(data: Dict[str, object]):
    return str(round(time.time()))


def save_hyperparams_yaml(hyperparams: Dict[str, object], fn: str):
    with open(fn, 'w') as hyperparamfile:
        yaml.dump(hyperparams, hyperparamfile, default_flow_style=False)


def load_models(agents, number_runners, model_path, model_name, iter):
    for agent in agents[:number_runners]:
        agent.load_model(os.path.join(model_path, model_name + '_iter_' + str(iter) + '_runner.pt'))
    for agent in agents[number_runners:]:
        agent.load_model(os.path.join(model_path, model_name + '_iter_' + str(iter) + '_waiting.pt'))


def plot_loss_and_reward_curves(curves_path, current_episode, losses, reward_sums, number_agents, model_name):
    # Compensate for different episode lengths
    losses_unitlength = []
    for a in range(number_agents):
        losses_unitlength += [[]]
        for e in range(current_episode):
            losses_unitlength[a] += [sum(losses[a][e]) / (1e-7 + len(losses[a][e]))]

    reward_sums_unitlength = []
    for a in range(number_agents):
        reward_sums_unitlength += [[]]
        for e in range(current_episode):
            reward_sums_unitlength[a] += [sum(reward_sums[a][e]) / (1e-7 + len(reward_sums[a][e]))]

    # Smoothing
    kernel_width = 5
    kernel = [1. / kernel_width for i in range(kernel_width)]  # Average over some episodes

    losses_smoothed = []
    for a in range(number_agents):
        losses_smoothed += [[]]
        for e in range(int((kernel_width - 1) / 2.), current_episode):
            losses_smoothed[a] += [sum([kernel[i] * losses_unitlength[a][e - i] for i in
                                        range(kernel_width)])]

    reward_sums_smoothed = []
    for a in range(number_agents):
        reward_sums_smoothed += [[]]
        for e in range(int((kernel_width - 1) / 2.), current_episode):
            reward_sums_smoothed[a] += [sum([kernel[i] * reward_sums_unitlength[a][e - i] for i in
                                             range(kernel_width)])]

    fig, axs = plt.subplots(1, 2, figsize=[20, 12])
    axs[0].set_title('Loss', size=12)
    axs[0].plot(numpy.array(losses_smoothed).transpose())
    # axs[0].set_ylim(-0.01, 1.5)  # Avoid that loss explosion destroys scale
    axs[1].set_title('Episode Reward', size=12)
    axs[1].plot(numpy.array(reward_sums_smoothed).transpose())
    try:
        plt.savefig(os.path.join(curves_path, 'loss_and_reward_curves_' + model_name + '.png'))
    except Exception as e:
        print(e)
    plt.close()


def plot_loss_and_reward_curves_averaged_over_agents(curves_path, current_episode, losses, reward_sums, number_agents, model_name):

    # Compensate for different episode lengths
    losses_unitlength = []
    reward_sums_unitlength = []
    for a in range(number_agents):
        losses_unitlength += [[]]
        reward_sums_unitlength += [[]]
        for e in range(current_episode):
            losses_unitlength[a] += [sum(losses[a][e]) / (1e-7 + len(losses[a][e]))]
            reward_sums_unitlength[a] += [sum(reward_sums[a][e]) / (1e-7 + len(reward_sums[a][e]))]

    losses_averaged_over_agents = losses_unitlength[1]
    reward_sums_averaged_over_agents = reward_sums_unitlength[1]
    for a in range(2, number_agents):
        for e in range(len(losses_unitlength[a])):
            losses_averaged_over_agents[e] += losses_unitlength[a][e]
            reward_sums_averaged_over_agents[e] += reward_sums_unitlength[a][e]
    losses_averaged_over_agents = [x / number_agents for x in losses_averaged_over_agents]
    reward_sums_averaged_over_agents = [x / number_agents for x in reward_sums_averaged_over_agents]

    fig, axs = plt.subplots(1, 2, figsize=[20, 12])
    axs[0].set_title('Loss', size=12)
    axs[0].plot(numpy.array(losses_averaged_over_agents).transpose())
    # axs[0].set_ylim(-0.01, 1.5)  # Avoid that loss explosion destroys scale
    axs[1].set_title('Episode Reward', size=12)
    axs[1].plot(numpy.array(reward_sums_averaged_over_agents).transpose())
    try:
        plt.savefig(os.path.join(curves_path, 'loss_and_reward_curves_averaged_over_agents' + model_name + '.png'))
    except Exception as e:
        print(e)
    plt.close()


def post_process(states: List[numpy.ndarray], episode_lengths: List[int], dt: float, trajectory_path: str,
                 model_name: str):
    runs = extract_trajectories(states, episode_lengths)
    plot_trajectories(runs)

    save_path = os.path.join(trajectory_path, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_trajectories(runs, episode_lengths, dt, save_path)


def readScenario(scenPath):
    with open(scenPath, 'r') as scenFile:
        scenario = scenFile.read()
    return scenario


def readTargetIDs():
    if not traci_store.target_ids:
        poly_ids = config.cli.poly.getIDList()
        target_ids = []
        for id in poly_ids:
            targetType = config.cli.poly.getType(id)
            if targetType == "TARGET":
                target_ids += [id]
        traci_store.target_ids = [x for x in target_ids if x != "6"]  # "6" is assumed to be the target of the runner
    return traci_store.target_ids


def readPersonIDList():
    if not traci_store.pers_id_list:
        traci_store.pers_id_list = list(config.cli.pers.getIDList())
    return traci_store.pers_id_list


def readTopographyBounds():
    if not traci_store.topography_bounds:
        traci_store.topography_bounds = config.cli.poly.getTopographyBounds()
    return traci_store.topography_bounds


def readTargetPositions():
    if not traci_store.target_positions:
        target_positions = []
        target_ids = readTargetIDs()
        for id in target_ids:
            target_positions += [config.cli.poly.getCentroid(id)]
        traci_store.target_positions = target_positions
    return traci_store.target_positions
