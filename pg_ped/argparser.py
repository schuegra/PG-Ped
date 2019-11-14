import argparse


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-mode', type=str, choices=['train', 'simulate', 'resumetraining', 'gridsearch'],
                        help='Wheter to train, simulate with given models, resume a training or do gridsearch for hyperparameter optimization.')
    parser.add_argument('-graphviz', type=bool, default=False,
                        help='Wheter to use graphviz.')
    parser.add_argument('-marltiming', type=str, choices=['sequential', 'onevsall'], default=['sequential'],
                        help='The scheme of optimization which is applied during training. Frequential means that after' \
                             'each episode, each policy is updated. Onevsall means that for some subsequent episodes,' \
                             'only one policy is updated.')
    parser.add_argument('-modelpath', type=str, default='states', help='Where models are stored and loaded from.')
    parser.add_argument('-modelname', type=str, default='experiment',
                        help='Name of the models.\nWhen saving or loading, an id is appended to this name for each agent.')
    parser.add_argument('-trajectorypath', type=str, default='trajectories', help='Where simulated trajectories are stored and loaded from.')
    parser.add_argument('-geometry', nargs=10, type=float,
                        default=[1., 4., 6., 3., 7., 7., 4.5, 8., 0.2, 0.25],
                        #default=[-3., 0., 1.55, 0., 2.0, 1.7, 0.775, 2., 0.15, 0.18],
                        #default=[-1., 0., 1.55, 0., 2.0, 1.7, 0.775, 2., 0.10, 0.15],
                        help='Experiment geometry: start_line xmin xmax ymin ymax goalline(y-coordinate) runner_goal_x runner_goal_y pedestrian_radius soft_pedestrian_radius')
    parser.add_argument('-steplengths', type=float, nargs=2, default=[1., 0.1],
                        help='Distance that runner or waiting move to their goals, if they can.')
    parser.add_argument('-pushdist', type=float, default=0.02,
                        help='Distance that a pushed pedestrian will be moved in a direction depending on the heuristic used.')
    parser.add_argument('-maximumspeed', type=float, nargs=2, default=[-0.3, 1.8],
                        help='Maximum speed of the pedestrians forward and backward.')
    parser.add_argument('-numberagents', type=int, default=1, help='Number of agents')
    parser.add_argument('-dt', type=float, default=0.5, help='Time interval per decision')
    parser.add_argument('-densitymap', nargs=2, type=float, default=[5., .1],
                        help='Density parameters: influence_radius(cut off parameter) standard_deviation')
    parser.add_argument('-heatmap', nargs=2, type=int, default=[45, 15], help='Heatmap parameters: rows columns')
    #parser.add_argument('-heatmap', nargs=2, type=int, default=[45, 15], help='Heatmap parameters: rows columns')
    parser.add_argument('-localwindow', nargs=2, type=int, default=[35, 35])
    #parser.add_argument('-localwindow', nargs=2, type=int, default=[35, 35])
    #parser.add_argument('-localwindow', nargs=2, type=int, default=[51, 51]) # both have to be odd
    parser.add_argument('-algorithmfloats', nargs=8, type=float, default=[0.99, 1e-4, 1e-4, 1e-0, 1e-0, 0.3, 0.001, 0.0],
                        help='Float hyperparameters to the learning algorithm: gamma, lr_runner_policy, lr_waiting_policy, \
                        lr_runner_value, lr_waiting_value, eps_start, eps_end, dropout_probability')
    #parser.add_argument('-algorithmints', nargs=10, type=int, default=[10000, 4, 3, 50000, 200, 1000000, 32, 32, 4, 10000],
    parser.add_argument('-algorithmints', nargs=10, type=int,
                        #default=[15000, 4, 3, 151, 150, 300000, 128, 32, 4, 10000],
                        default=[1, 4, 3, 301, 300, 300000, 128, 32, 4, 10000],
                        help='Int hyperparameters to the learning algorithm: nr_episodes, state_variables_per_agent_per_timestep, \
                        number_timesteps_per_state, memorization_capacity_of_agent, maximum_number_of_steps, eps_decay_length, \
                        numberofsamples, batchsize, optimization_frequency, start_learning')
    parser.add_argument('-simulationruns', type=int, default=5,
                        help='Number of simulation runs which will also be animated.')
    parser.add_argument('-randomseed', type=int, default=0,
                        help='Random seed for reproducebility.')
    parser.add_argument('-optimizermessagefrequency', type=int, default=5,
                        help='How often to print training progress to the console.')
    parser.add_argument('-mlp', nargs=3, default=[1, [128], 'tanh'],
                        help='Multi layer perceptron with: n_hiddens n_neurons activation')
    parser.add_argument('-scenPath', type=str, help='Path to the scenario', default='/Users/Philipp/Downloads/scenarios/')
    parser.add_argument('-scenFName', type=str, help='Filename of the scenario. E.g. scenario002.scenario', default='denseCrowd.scenario')
    return parser


def parse_args(args, parser):
    parsed_args = parser.parse_args(args[1:])
    return parsed_args
