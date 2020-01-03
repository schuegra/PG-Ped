import os
import time
import yaml
from typing import Dict, List
from copy import deepcopy

import numpy
import matplotlib.pyplot as plt

from pg_ped.visualization.plot_trajectories import plot_trajectories
from pg_ped.evaluation.evaluate_trajectories import extract_trajectories, save_trajectories


def create_fn(data: Dict[str, object]):
    return str(round(time.time()))


def save_hyperparams_yaml(hyperparams: Dict[str, object], fn: str):
    with open(fn, 'w') as hyperparamfile:
        yaml.dump(hyperparams, hyperparamfile, default_flow_style=False)


def load_models(agents, number_runners, model_path, model_name):
    for agent in agents[:number_runners]:
        agent.load_model(os.path.join(model_path, model_name + '_iter_50_runner.pt'))
    for agent in agents[number_runners:]:
        agent.load_model(os.path.join(model_path, model_name + '_iter_50_waiting.pt'))

def plot_loss_and_reward_curves(current_episode, losses, reward_sums, number_agents, model_name):
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
    kernel = [1./kernel_width for i in range(kernel_width)]  # Average over some episodes

    losses_smoothed = []
    for a in range(number_agents):
        losses_smoothed += [[]]
        for e in range(int((kernel_width - 1)/2.), current_episode):
            losses_smoothed[a] += [sum([kernel[i] * losses_unitlength[a][e - i] for i in
                                        range(kernel_width)])]

    reward_sums_smoothed = []
    for a in range(number_agents):
        reward_sums_smoothed += [[]]
        for e in range(int((kernel_width - 1)/2.), current_episode):
            reward_sums_smoothed[a] += [sum([kernel[i] * reward_sums_unitlength[a][e - i] for i in
                                        range(kernel_width)])]


    fig, axs = plt.subplots(1, 2, figsize=[20, 12])
    axs[0].set_title('Loss', size=12)
    axs[0].plot(numpy.array(losses_smoothed).transpose())
    #axs[0].set_ylim(-0.01, 1.5)  # Avoid that loss explosion destroys scale
    axs[1].set_title('Episode Reward', size=12)
    axs[1].plot(numpy.array(reward_sums_smoothed).transpose())
    plt.savefig(os.path.join('curves', 'loss_and_reward_curves_' + model_name + '.png'))
    plt.close()

def post_process(states: List[numpy.ndarray], episode_lengths: List[int], dt: float, trajectory_path: str, model_name: str):

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