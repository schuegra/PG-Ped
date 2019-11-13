from typing import List, Callable, Tuple, Union
from math import ceil

import numpy
import matplotlib.pyplot as plt



def plot_trajectories(runs: Union[numpy.ndarray, List[numpy.ndarray]]):
    '''
        Plots the trajectories in each run
        runs has shape n_runs 2
    '''

    numpy.random.seed(0)

    if isinstance(runs[0], List):
        n_agents = len(runs)
        root_n_agents = n_agents ** 0.5
        cols = int(ceil(root_n_agents))
        rows = int(root_n_agents)
        fig, axs = plt.subplots(rows, cols, figsize=[12, 12])
        for i, agent in enumerate(runs):
            r, c = int(i / cols), i % cols
            color = numpy.random.rand(3,)
            for run in agent:
                positions = numpy.array([state[0, :2] for state in run])
                axs[r, c].plot(positions[:, 0], positions[:, 1], color=color)
            axs[r, c].axis('off')
            axs[r, c].set_aspect('equal')

    else:
        fig, ax = plt.subplots()
        for run in runs:
            positions = numpy.array([state[0, :2] for state in run])
            ax.plot(positions[:, 0], positions[:, 1])
        ax.axis('off')
        ax.set_aspect('equal')
    plt.show()