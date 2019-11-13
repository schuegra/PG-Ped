import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as colors
from matplotlib.patches import Circle
import numpy as np

from pg_ped.marl.utils import cart_to_img_numpy
from pg_ped.environment_construction.geometry import update_state_without_collision_check


def vis_reward(reward_func, state, agent_identity, initial_state,
               x_min, x_max, y_min, y_max, start_line, goal_line,
               fig=None, ax=None, **kwargs):
    state_before = state.clone()
    del state

    x_res = 15
    y_res = int(x_res * (y_max - start_line) / (x_max - x_min))
    X, Y = np.linspace(x_min, x_max, x_res), np.linspace(start_line, y_max, y_res)
    # X, Y = np.meshgrid(X, Y)

    R = np.zeros([y_res + 1, x_res + 1])
    for x in X:
        for y in Y:
            u = int(y_res * (1 - ((y - start_line) / (y_max - start_line))))
            v = int(x_res * (x - x_min) / (x_max - x_min))
            temp_state = update_state_without_collision_check(state_before, x, y, agent_identity,
                                                              x_min=x_min, x_max=x_max, y_min=start_line,
                                                              y_max=y_max, eps=1e-8, **kwargs)
            kwargs['initial_state'] = initial_state
            R[u, v] = reward_func(temp_state, agent_identity,
                                  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                  start_line=start_line, goal_line=goal_line, **kwargs)

    R[y_res - 1, :] = R[y_res, :]
    R[:, x_res - 1] = R[:, x_res]
    R = R[:-1, :-1]

    # Plot
    if not ax:
        fig, ax = plt.subplots()
    else:
        ax.clear()
    ax.imshow(np.log(R - R.min() + 1e-8), cmap='gist_gray')
    ax.imshow(R, cmap='gist_gray')
    positions = state_before[:, :2].cpu().numpy()
    for i, p in enumerate(positions):
        x, y = p[0], p[1]
        u = int(y_res * (1 - ((y - start_line) / (y_max - start_line))))
        v = int(x_res * (x - x_min) / (x_max - x_min))
        circ = Circle((v, u), 0.5)
        ax.add_patch(circ)
    ax.axis('off')
    #plt.colorbar()
    #plt.show()


class Animator(object):

    def __init__(self,
                 reward_func,
                 frames,
                 agent_identity,
                 visualizer,
                 initial_state,
                 **kwargs) -> None:
        self._initial_state = initial_state
        self._reward_func = reward_func
        self._frames = frames
        self._agent_identity = agent_identity
        self._visualizer = visualizer
        np.random.seed(0)

        self._kwargs = kwargs

    def animate(self, animation_name='animation', frames_per_second=3):
        # These frames will be shown in the animation
        frames_to_animate = self._frames

        # initialization function: plot the initial frame
        fig, ax = plt.subplots()

        def init():
            ax.axis('off')
            self._visualizer(self._reward_func, self._frames[0], self._agent_identity, self._initial_state,
                             ax=ax, **self._kwargs)
            return ax,

        # animation function. This is called sequentially
        def animate(i):
            self._visualizer(self._reward_func, self._frames[i], self._agent_identity, self._initial_state,
                             ax=ax, **self._kwargs)
            return ax,

        # call the animator.
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       init_func=init,
                                       frames=len(frames_to_animate),
                                       interval=40)

        # save the animation as an mp4
        anim.save(animation_name + '.mp4', fps=frames_per_second)
