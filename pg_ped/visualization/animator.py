from typing import List, Callable, Tuple

import numpy
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from torch import Tensor
import torch


def init_axes(ax, x_min, x_max, y_min, y_max, goal_line,
              color_wall='r', color_goal='b', start_line=0., **kwargs) -> None:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(start_line, y_max)
    ax.hlines([start_line, y_min, goal_line, y_max], xmin=x_min, xmax=x_max, color=color_goal)
    ax.vlines([x_min, x_max], ymin=start_line, ymax=y_max, color=color_wall)
    ax.set_aspect('equal')
    ax.axis('off')


def vis_state_wrapper(positions: Tensor, person_radius: float, *args) -> None:
    numpy.random.seed(0)
    color_runner = numpy.random.rand()
    color_waiting = numpy.random.rand()
    colors = numpy.array([color_runner] + [color_waiting for i in range(positions.shape[0])])
    fig, ax = plt.subplots()

    vis_state(positions, colors, ax, fig, person_radius, 0, 1.5, 0, 2, 1.8)


def vis_state(state: Tensor, colors: numpy.ndarray, ax, fig, person_radius: float, soft_person_radius: float,
              x_min: float, x_max: float, y_min: float, y_max: float, goal_line: float,
              variables_per_agent_per_timestep: int, backward_view: int,
              rows: int, cols: int, device: str, standard_deviation: float,
              with_arrows=False, with_init_axes=True, **kwargs) -> None:
    '''
        args:
            positions - a tensor of shape (n,2)
    '''
    positions = state[:, :2]
    directions = 0.2 * numpy.vstack([state[:, 2], state[:, 3]]).T
    circles = [Circle(xy, radius=person_radius, fill=False) for xy in positions]
    circles2 = [Circle(xy, radius=soft_person_radius, fill=False) for xy in positions]

    ax.clear()
    init_axes(ax, x_min, x_max, y_min, y_max, goal_line, **kwargs)
    p = PatchCollection(circles)
    p2 = PatchCollection(circles2)
    p.set_array(colors)
    ax.add_collection(p2)
    ax.add_collection(p)
    if with_arrows is True:
        for i in range(positions.shape[0]):
            ax.arrow(positions[i, 0], positions[i, 1], directions[i, 0], directions[i, 1], head_width=0.1)


class Animator(object):

    def __init__(self, frames: List[Tensor],
                 visualizer: Callable[[object], None],
                 **kwargs) -> None:
        self._frames = frames
        self._visualizer = visualizer
        numpy.random.seed(0)
        color_runner = numpy.random.rand()
        color_waiting = numpy.random.rand()
        self._colors = numpy.array([color_runner] + [color_waiting for i in range(frames[0].shape[0])])
        self._kwargs = kwargs

    def animate(self, animation_name='animation', frames_per_second=3):
        # These frames will be shown in the animation
        frames_to_animate = self._frames

        # initialization function: plot the initial frame
        fig, ax = plt.subplots()

        def init():
            init_axes(ax, **self._kwargs)
            self._visualizer(self._frames[0], self._colors, ax, fig, **self._kwargs)
            return ax,

        # animation function. This is called sequentially
        def animate(i):
            self._visualizer(self._frames[i], self._colors, ax, fig, **self._kwargs)
            return ax,

        # call the animator.
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       init_func=init,
                                       frames=len(frames_to_animate),
                                       interval=40)

        # save the animation as an mp4
        anim.save(animation_name + '.mp4', fps=frames_per_second)
