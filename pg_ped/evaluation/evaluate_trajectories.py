from typing import List, Callable, Tuple
import os

import numpy
import matplotlib.pyplot as plt


def extract_trajectories(states: List[numpy.ndarray], episode_lengths: List[int]):
    # Split states to episodes
    split_indices = [sum(episode_lengths[:i + 1]) for i in range(len(episode_lengths))][:-1]
    # split_indices = [si if si < len(states) else len(states) - 1 for si in split_indices]
    runs = numpy.split(states, split_indices)

    return runs


def save_trajectories(runs: List, episode_lengths: List[int], dt: float, path: str):

    episode_durations = [el * dt for el in episode_lengths]
    for i, run in enumerate(runs):
        t = numpy.expand_dims(numpy.arange(0, episode_durations[i], dt), 1)
        for a in range(run.shape[1]):
            pv = numpy.array([state[a, :4] for state in run])
            pv = numpy.hstack([t, pv])

            episode_path = os.path.join(path, 'episode_' + str(i))
            if not os.path.exists(episode_path):
                os.mkdir(episode_path)
            numpy.savetxt(os.path.join(episode_path, 'agent_' + str(a) + '.txt'), pv,
                          delimiter=',', header='t (s), x (m), y (m), orientation_x, orientation_y')
