import os
from datetime import datetime
import pytz
import gc

from pg_ped.helpers import plot_loss_and_reward_curves, create_fn, save_hyperparams_yaml
from pg_ped.utils import set_state_vadere

def start_training(episodes, parameter_dict, losses, reward_sums, episode_lengths, number_episodes, number_agents,
                   model_name, model_path, optimizer_message_frequency, plot=True, use_traci=False, ports=None):

    tz = pytz.timezone('Europe/Berlin')
    berlin_now = datetime.now(tz).isoformat()

    print('****************START TRAINING AT ', berlin_now, '****************\n')
    if ports:
        print('Ports (vadere, java, python)', ports[0], ports[1], ports[2])
    for i, episode in enumerate(episodes):

        if use_traci is True:
            set_state_vadere(episode._single_agent_steps[0]._environment._initial_state)

        parameter_dict['current_episode'] = i
        done = False
        failed = False
        j = 0
        while done is False and failed is False:
            parameter_dict['current_step'] = j
            state, done, failed = episode(**parameter_dict)
            j += 1
        episode_lengths += [j]
        gc.collect() # clean up as much as possible after each episode

        if i % 5 == 0:
            print('****************EPISODES TRAINED ' + str(i + 1) + '****************\n')

        if i > 0:
            if i + 1 % 100 == 0:
                episode.save_models(os.path.join(model_path, model_name + '_iter_' + str(i + 1)), **parameter_dict)
            if i % 1 == 0:
                plot_loss_and_reward_curves(i, losses, reward_sums, number_agents, model_name)


    episode.save_models(os.path.join(model_path, model_name + '_iter_' + str(i + 1)), **parameter_dict)

    if plot is True:# Visualize loss and reward
        plot_loss_and_reward_curves(number_episodes, losses, reward_sums, number_agents, model_name)

    berlin_now = datetime.now(tz).isoformat()
    print('****************FINISHED TRAINING AT ', berlin_now, '****************\n')

