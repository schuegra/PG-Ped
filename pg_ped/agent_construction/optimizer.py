from typing import List, Callable

import numpy

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from pg_ped.utils import normalize_tensor, prepare, standardize_tensor
from pg_ped.environment_construction.state_representation import (render_vision_field_torch,
                                                                  generate_kinematics_torch,
                                                                  generate_kinematics_torch2)
from pg_ped.visualization.visualize_cnn import vis_feature_maps
from pg_ped.environment_construction.termination import runners_in_goal
from pg_ped.agent_construction.action_selector import continous_action_selection_normal

try:
    from pg_ped.visualization.graph_visualization import make_dot
except ImportError:
    pass


def episode_discounted_rewards(rewards: List[Tensor], discount_factor: float) -> List[Tensor]:
    R = 0
    discounted_rewards = []

    # Discount future rewards back to the present using gamma
    for r in rewards[::-1]:
        R = r + discount_factor * R
        discounted_rewards.insert(0, R)

    discounted_rewards = numpy.array(discounted_rewards)
    mean = discounted_rewards.mean()
    std = discounted_rewards.std()
    discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

    return discounted_rewards


def discount_rewards(rewards, gamma):
    r = numpy.array([gamma ** i * rewards[i]
                     for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r  # (r - r.mean()) # / (1e-7 + r.std())


def optimize(policy: nn.Module,
             dummy1,
             episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
             optimizer_policy: object,
             dummy2,
             states: List[Tensor],
             actions: List[Tensor],
             action_indicees: List[Tensor],
             log_probs: List[Tensor],
             probs: List[Tensor],
             next_states: List[Tensor],
             rewards: List[Tensor],
             state_values: List[Tensor],
             next_state_values: List[Tensor],
             agent_identity: int,
             discount_factor: float,
             reward_sums: List[List[float]],
             losses: List[List[float]],
             current_episode: int,
             current_step: int,
             optimizer_message_frequency: int,
             learning_rate_runner_policy: float,
             learning_rate_waiting_policy: float,
             learning_rate_runner_value: float,
             learning_rate_waiting_value: float,
             device: str,
             episode_lengths: List[int],
             **kwargs) -> None:
    reward_sums[agent_identity][current_episode].append(sum(rewards) / len(rewards))

    discounted_rewards = episode_discounted_rewards(rewards, discount_factor)
    # discounted_rewards = numpy.array(rewards).sum()
    discounted_rewards = torch.tensor(discounted_rewards, device=device).float()
    discounted_rewards.requires_grad = True

    # probs = torch.cat(probs).squeeze(1).detach()
    log_probs = torch.cat(log_probs).squeeze(1)  # .to(device) # alternatively use sum
    loss = torch.mean(- log_probs * discounted_rewards)

    # loss.register_hook(print)
    # log_probs.register_hook(print)
    # entropies = Variable(torch.mean(- log_probs * probs), requires_grad=True)
    # loss = loss - 0.1 * entropies

    losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

    # Optimize policy
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()

    number_of_time_steps_since_training_start = sum(episode_lengths) + current_step
    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float(loss.detach().cpu().numpy()), 8))

    del log_probs, probs, rewards, loss, discounted_rewards


def optimize_1step_td(policy: nn.Module,
                      value: nn.Module,
                      episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                      optimizer_policy: object,
                      optimizer_value: object,
                      states: List[Tensor],
                      actions: List[Tensor],
                      action_indicees: List[Tensor],
                      log_probs: List[Tensor],
                      probs: List[Tensor],
                      next_states: List[Tensor],
                      rewards: List[Tensor],
                      state_values: List[Tensor],
                      next_state_values: List[Tensor],
                      agent_identity: int,
                      discount_factor: float,
                      reward_sums: List[List[float]],
                      losses: List[List[float]],
                      current_episode: int,
                      current_step: int,
                      optimizer_message_frequency: int,
                      learning_rate_runner_policy: float,
                      learning_rate_waiting_policy: float,
                      learning_rate_runner_value: float,
                      learning_rate_waiting_value: float,
                      device: str,
                      **kwargs) -> None:
    if agent_identity == 0:
        lr_policy = learning_rate_runner_policy
        lr_value = learning_rate_runner_value
    else:
        lr_policy = learning_rate_waiting_policy
        lr_value = learning_rate_waiting_value

    discounted_rewards = [discount_factor * next_state_values[i].detach() + rewards[i] for i in range(len(rewards))]
    discounted_rewards = torch.cat(discounted_rewards)

    advantages = torch.cat([discounted_rewards[i] - state_values[i]
                            for i in range(len(state_values))])
    # entropies = [- log_probs[i] * torch.tensor(probs[i], device=device).float() for i in range(len(probs))]
    # entropies = Variable(torch.mean(torch.cat(entropies)).detach(), requires_grad=True)
    log_probs = torch.cat(log_probs)
    entropies = - log_probs * probs[0]
    entropies = Variable(entropies.detach(), requires_grad=True)

    value_loss = 0.5 * advantages[0] ** 2
    policy_loss = - (discount_factor ** current_step) * log_probs[0] * Variable(advantages[0].detach(),
                                                                                requires_grad=True)  # - 1e-2 * entropies

    loss = lr_value * value_loss + lr_policy * policy_loss
    losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())
    reward_sums[agent_identity][current_episode].append(sum(rewards))

    if current_episode % optimizer_message_frequency == 0:
        print('LOSS (policy, value), REWARD AGENT', str(agent_identity), ': ',
              round(float(policy_loss), 8), round(float(value_loss), 8), ', ', float(sum(rewards)))

    # Optimize policy
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    del advantages, entropies, log_probs, rewards, value_loss, policy_loss


def optimize_1step_td_shared_optimizer(policy: nn.Module,
                                       value: nn.Module,
                                       episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                                       optimizer_policy: object,
                                       optimizer_value: object,
                                       states: List[Tensor],
                                       actions: List[Tensor],
                                       action_indicees: List[Tensor],
                                       log_probs: List[Tensor],
                                       probs: List[Tensor],
                                       next_states: List[Tensor],
                                       rewards: List[Tensor],
                                       state_values: List[Tensor],
                                       next_state_values: List[Tensor],
                                       agent_identity: int,
                                       discount_factor: float,
                                       reward_sums: List[List[float]],
                                       losses: List[List[float]],
                                       current_episode: int,
                                       current_step: int,
                                       optimizer_message_frequency: int,
                                       device: str,
                                       **kwargs) -> None:
    discounted_rewards = [rewards[i] + discount_factor * next_state_values[i].detach() for i in range(len(rewards))]
    discounted_rewards = torch.cat(discounted_rewards)

    advantages = torch.cat([discounted_rewards[i] - state_values[i]
                            for i in range(len(state_values))])
    # entropies = [- log_probs[i] * torch.tensor(probs[i], device=device).float() for i in range(len(probs))]
    # entropies = Variable(torch.mean(torch.cat(entropies)).detach(), requires_grad=True)
    log_probs = torch.cat(log_probs)
    entropies = - log_probs * probs[0]
    entropies = Variable(entropies.detach(), requires_grad=True)

    value_loss = 0.5 * advantages[0].pow(2)
    policy_loss = - log_probs[0] * Variable(advantages[0].detach(), requires_grad=True)

    loss = 0.5 * value_loss + policy_loss  # - 1e-2 * entropies
    losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())
    reward_sums[agent_identity][current_episode].append(sum(rewards))

    if current_episode % optimizer_message_frequency == 0:
        print('LOSS (policy, value), REWARD AGENT', str(agent_identity), ': ',
              round(float(policy_loss), 8), round(float(value_loss), 8), ', ', float(sum(rewards)))

    # Optimize policy
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()

    del advantages, entropies, log_probs, rewards, value_loss, policy_loss


def optimize_monte_carlo_shared_optimizer(policy: nn.Module,
                                          value: nn.Module,
                                          episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                                          optimizer_policy: object,
                                          optimizer_value: object,
                                          states: List[Tensor],
                                          actions: List[Tensor],
                                          action_indicees: List[Tensor],
                                          log_probs: List[Tensor],
                                          probs: List[Tensor],
                                          next_states: List[Tensor],
                                          rewards: List[Tensor],
                                          state_values: List[Tensor],
                                          next_state_values: List[Tensor],
                                          agent_identity: int,
                                          discount_factor: float,
                                          reward_sums: List[List[float]],
                                          losses: List[List[float]],
                                          current_episode: int,
                                          current_step: int,
                                          optimizer_message_frequency: int,
                                          device: str,
                                          **kwargs) -> None:
    reward_sums[agent_identity] += [sum(rewards)]

    discounted_rewards = discount_rewards(rewards, discount_factor)
    discounted_rewards = torch.tensor(discounted_rewards, device=device)
    discounted_rewards = Variable(discounted_rewards, requires_grad=True).float()

    # discounted_rewards = [rewards[i] + discount_factor*next_state_values[i] for i in range(len(rewards))]
    # discounted_rewards = torch.tensor(discounted_rewards, device=device)
    # discounted_rewards = (discounted_rewards - discounted_rewards.mean())/(1e-7 + discounted_rewards.std())

    advantages = torch.cat([discounted_rewards[i] - state_values[i] for i in range(len(state_values))])
    entropies = [- log_probs[i] * torch.tensor(probs[i], device=device).float() for i in range(len(probs))]
    entropies = Variable(torch.mean(torch.cat(entropies)).detach(), requires_grad=True)
    log_probs = torch.cat(log_probs)

    value_loss = torch.mul(torch.mean(torch.pow(advantages, 2)), 0.5)
    policy_loss = torch.mean(torch.mul(log_probs, -1) * Variable(advantages.detach(), requires_grad=True))

    loss = value_loss.mul(0.5) + policy_loss + entropies.mul(- 0.0001)
    losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

    if current_episode % optimizer_message_frequency == 0:
        print('LOSS (policy, value), REWARD AGENT', str(agent_identity), ': ',
              round(float(policy_loss), 4), round(float(value_loss), 4), ', ', float(sum(rewards)))

    # Optimize policy
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()

    del advantages, entropies, log_probs, rewards, value_loss, policy_loss


def optimize_Q(policy: nn.Module,
               episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
               optimizer_policy: object,
               states: List[Tensor],
               actions: List[Tensor],
               action_indicees: List[Tensor],
               log_probs: List[Tensor],
               probs: List[Tensor],
               next_states: List[Tensor],
               rewards: List[Tensor],
               state_values: List[Tensor],
               next_state_values: List[Tensor],
               agent_identity: int,
               discount_factor: float,
               reward_sums: List[List[float]],
               losses: List[List[float]],
               current_episode: int,
               current_step: int,
               optimizer_message_frequency: int,
               learning_rate_runner_policy: float,
               learning_rate_waiting_policy: float,
               learning_rate_runner_value: float,
               learning_rate_waiting_value: float,
               device: str,
               **kwargs) -> None:
    reward_sums[agent_identity][current_episode].append(sum(rewards) / len(rewards))

    probs = [prob.to(device) for prob in probs]
    discounted_rewards = episode_discounted_rewards(rewards, discount_factor)
    discounted_rewards = torch.tensor(discounted_rewards, device=device)
    discounted_rewards = Variable(discounted_rewards, requires_grad=True).float()
    advantages = torch.cat([discounted_rewards[i] - probs[i] for i in range(len(state_values))])  # use probs as qvalues
    loss = torch.mul(torch.mean(torch.pow(advantages, 2)), 0.5)

    losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

    # Optimize policy
    optimizer_policy.zero_grad()
    loss.backward()
    optimizer_policy.step()

    if current_episode % optimizer_message_frequency == 0:
        print('LOSS, DISC. REWARD, REWARD AGENT', str(agent_identity), ': ',
              round(float(loss.detach().cpu().numpy()), 8), ', ',
              round(float(discounted_rewards[0].detach().cpu().numpy()), 8),
              ', ', float(sum(rewards)))

    del log_probs, probs, rewards, loss, discounted_rewards


def transform_states(states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs):
    scenarios = []
    for state in states:
        scenarios += [
            render_vision_field_torch(state.to(device), agent_identity, device=device,
                                      runner_identities=runner_identities, goal_line=goal_line,
                                      person_radius=person_radius, **kwargs)]
    return scenarios


def transform_states_to_kinematics(states, agent_identity, device, runner_identities, goal_line, person_radius,
                                   **kwargs):
    kinematics = []
    for state in states:
        kinematics += [
            generate_kinematics_torch(state.to(device), agent_identity, device=device,
                                      runner_identities=runner_identities, goal_line=goal_line,
                                      person_radius=person_radius, **kwargs)]
    return scenarios


def optimize_QExperienceReplay(value_net: nn.Module,
                               value_net2: nn.Module,
                               episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                               optimizer: object,
                               optimizer_value: object,
                               states: List[Tensor],
                               actions: List[Tensor],
                               action_indicees: List[Tensor],
                               log_probs: List[Tensor],
                               probs: List[Tensor],
                               next_states: List[Tensor],
                               rewards: List[Tensor],
                               state_values: List[Tensor],
                               next_state_values: List[Tensor],
                               agent_identity: int,
                               discount_factor: float,
                               reward_sums: List[List[float]],
                               losses: List[List[float]],
                               current_episode: int,
                               current_step: int,
                               optimizer_message_frequency: int,
                               learning_rate_runner_policy: float,
                               learning_rate_waiting_policy: float,
                               learning_rate_runner_value: float,
                               learning_rate_waiting_value: float,
                               device: str,
                               number_samples: int,
                               batch_size: int,
                               runner_identities: List[int],
                               goal_line: int,
                               person_radius: float,
                               **kwargs) -> None:
    # reward_sums[agent_identity][current_episode].append(sum(rewards))

    states_on_device = torch.cat(
        transform_states(states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs))
    next_states_on_device = torch.cat(
        transform_states(next_states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs))

    all_rewards = torch.tensor(rewards, device=device).float()
    # all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    number_steps = int(states_on_device.shape[0] / batch_size + 1)
    value_net.train()
    for i in range(number_steps):
        start = batch_size * i
        end = batch_size * (i + 1)
        end = min(end, states_on_device.shape[0])
        if start >= end:
            break

        s0 = states_on_device.data[start: end]
        s1 = next_states_on_device.data[start: end]
        indices = ([torch.cat([torch.tensor([i for i in range(end - start)])]).unsqueeze(0),
                    torch.cat(action_indicees[start: end]).unsqueeze(0)])
        q0 = value_net(s0)[indices[0], indices[1]]
        next_state_q = value_net2(s1)

        # Check terminal states and set their values to zero
        if agent_identity in runner_identities:
            for j, state in enumerate(next_states[start: end]):
                for id in runner_identities:
                    if runners_in_goal(state, id, runner_identities, goal_line, person_radius, **kwargs) is True:
                        next_state_q[j] = next_state_q[j, :] * 0.

        q1 = torch.max(next_state_q, dim=1)[0].unsqueeze(0).detach()
        r = all_rewards[start: end]

        target_q = Variable(r + discount_factor * q1, requires_grad=True)
        loss = value_net.loss(q0, target_q)

        # bellmann_error = q0 - target_q
        # loss = 0.5 * torch.mean(bellmann_error ** 2)

        losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

        # Optimize policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target net
    number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
    if number_of_time_steps_since_training_start % int(1e4) == 0:
        value_net2.load_state_dict(value_net.state_dict())

    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float(loss.detach().cpu().numpy()), 8))

    del s0, s1, indices, q0, q1, r, loss, states_on_device, next_states_on_device, target_q, next_state_q  # , bellmann_error


def transform_states_KINEMATICS(states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs):
    scenarios = []
    for state in states:
        scenarios += [
            # generate_kinematics_torch(
            generate_kinematics_torch2(
                state.to(device), agent_identity, device=device,
                runner_identities=runner_identities, goal_line=goal_line,
                person_radius=person_radius, **kwargs
            )
        ]
    return scenarios


def optimize_QExperienceReplay_KINEMATICS(value_net: nn.Module,
                                          value_net2: nn.Module,
                                          episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                                          optimizer: object,
                                          optimizer_value: object,
                                          states: List[Tensor],
                                          actions: List[Tensor],
                                          action_indicees: List[Tensor],
                                          log_probs: List[Tensor],
                                          probs: List[Tensor],
                                          next_states: List[Tensor],
                                          rewards: List[Tensor],
                                          state_values: List[Tensor],
                                          next_state_values: List[Tensor],
                                          agent_identity: int,
                                          discount_factor: float,
                                          reward_sums: List[List[float]],
                                          losses: List[List[float]],
                                          current_episode: int,
                                          current_step: int,
                                          optimizer_message_frequency: int,
                                          learning_rate_runner_policy: float,
                                          learning_rate_waiting_policy: float,
                                          learning_rate_runner_value: float,
                                          learning_rate_waiting_value: float,
                                          device: str,
                                          number_samples: int,
                                          batch_size: int,
                                          runner_identities: List[int],
                                          goal_line: int,
                                          person_radius: float,
                                          **kwargs) -> None:
    # reward_sums[agent_identity][current_episode].append(sum(rewards))

    states_on_device = torch.cat(states).to(device)
    next_states_on_device = torch.cat(next_states).to(device)

    all_rewards = torch.tensor(rewards, device=device).float()
    # all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    number_steps = int(states_on_device.shape[0] / batch_size + 1)
    value_net.train()
    for i in range(number_steps):
        start = batch_size * i
        end = batch_size * (i + 1)
        end = min(end, states_on_device.shape[0])
        if start >= end:
            break

        s0 = states_on_device.data[start: end]
        s1 = next_states_on_device.data[start: end]
        indices = ([torch.cat([torch.tensor([i for i in range(end - start)])]).unsqueeze(0),
                    torch.cat(action_indicees[start: end]).unsqueeze(0)])
        q0 = value_net(s0)[indices[0], indices[1]]
        next_state_q = value_net2(s1)

        # Check terminal states and set their values to zero
        if agent_identity in runner_identities:
            for j, state in enumerate(next_states[start: end]):
                for id in runner_identities:
                    if runners_in_goal(state, id, runner_identities, goal_line, person_radius, **kwargs) is True:
                        next_state_q[j] = next_state_q[j, :] * 0.

        q1 = torch.max(next_state_q, dim=1)[0].unsqueeze(0).detach()
        r = all_rewards[start: end]

        target_q = Variable(r + discount_factor * q1, requires_grad=True)
        loss = value_net.loss(q0, target_q)

        # bellmann_error = q0 - target_q
        # loss = 0.5 * torch.mean(bellmann_error ** 2)

        losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

        # Optimize policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target net
    number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
    if number_of_time_steps_since_training_start % int(1e5) == 0:
        value_net2.load_state_dict(value_net.state_dict())

    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float(loss.detach().cpu().numpy()), 8))

    del s0, s1, indices, q0, q1, r, loss, states_on_device, next_states_on_device, target_q, next_state_q  # , bellmann_error


def optimize_QExperienceReplay_KINEMATICS_thompsonsampling(value_net: nn.Module,
                                                           value_net2: nn.Module,
                                                           episode_discounted_rewards: Callable[
                                                               [List[Tensor], float], Tensor],
                                                           optimizer: object,
                                                           optimizer_value: object,
                                                           states: List[Tensor],
                                                           actions: List[Tensor],
                                                           action_indicees: List[Tensor],
                                                           log_probs: List[Tensor],
                                                           values: List[Tensor],
                                                           next_states: List[Tensor],
                                                           rewards: List[Tensor],
                                                           state_values: List[Tensor],
                                                           next_state_values: List[Tensor],
                                                           agent_identity: int,
                                                           discount_factor: float,
                                                           reward_sums: List[List[float]],
                                                           losses: List[List[float]],
                                                           current_episode: int,
                                                           current_step: int,
                                                           optimizer_message_frequency: int,
                                                           learning_rate_runner_policy: float,
                                                           learning_rate_waiting_policy: float,
                                                           learning_rate_runner_value: float,
                                                           learning_rate_waiting_value: float,
                                                           device: str,
                                                           number_samples: int,
                                                           batch_size: int,
                                                           runner_identities: List[int],
                                                           goal_line: int,
                                                           person_radius: float,
                                                           **kwargs) -> None:
    # reward_sums[agent_identity][current_episode].append(sum(rewards))

    states_on_device = torch.cat(states).to(device)
    next_states_on_device = torch.cat(next_states).to(device)
    # values_on_device = torch.cat(values) #.to(device)
    # next_state_values_on_device = torch.cat(next_state_values).to(device)

    all_rewards = torch.tensor(rewards, device=device).float()
    all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    number_steps = int(states_on_device.shape[0] / batch_size + 1)
    value_net.train()
    for i in range(number_steps):
        start = batch_size * i
        end = batch_size * (i + 1)
        end = min(end, states_on_device.shape[0])
        if start >= end:
            break

        s0 = states_on_device.data[start: end]
        s1 = next_states_on_device.data[start: end]
        indices = ([torch.cat([torch.tensor([i for i in range(end - start)])]).unsqueeze(0),
                    torch.cat(action_indicees[start: end]).unsqueeze(0)])
        q0 = value_net(s0)[indices[0], indices[1]]
        # q0 = values_on_device[start: end].unsqueeze(0)
        next_state_q = value_net2(s1)

        # Check terminal states and set their values to zero
        if agent_identity in runner_identities:
            for j, state in enumerate(next_states[start: end]):
                for id in runner_identities:
                    if runners_in_goal(state, id, runner_identities, goal_line, person_radius, **kwargs) is True:
                        next_state_q[j] = next_state_q[j] * 0.

        q1 = torch.max(next_state_q, dim=1)[0].unsqueeze(0).detach()
        r = all_rewards[start: end]

        target_q = Variable(r + discount_factor * q1, requires_grad=True)
        loss = value_net.loss(q0, target_q)

        # bellmann_error = q0 - target_q
        # loss = 0.5 * torch.mean(bellmann_error ** 2)

        losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

        # Optimize policy
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # Update target net
    number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
    if number_of_time_steps_since_training_start % 100 == 0:
        value_net2.load_state_dict(value_net.state_dict())

    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float(loss.detach().cpu().numpy()), 8))

    del s0, s1, indices, q0, q1, r, loss, states_on_device, next_states_on_device, target_q, next_state_q  # , bellmann_error


def optimize_Qtd(policy: nn.Module,
                 value: nn.Module,
                 episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                 optimizer: object,
                 optimizer_value: object,
                 states: List[Tensor],
                 actions: List[Tensor],
                 action_indicees: List[Tensor],
                 log_probs: List[Tensor],
                 probs: List[Tensor],
                 next_states: List[Tensor],
                 rewards: List[Tensor],
                 state_values: List[Tensor],
                 next_state_values: List[Tensor],
                 agent_identity: int,
                 discount_factor: float,
                 reward_sums: List[List[float]],
                 losses: List[List[float]],
                 current_episode: int,
                 current_step: int,
                 optimizer_message_frequency: int,
                 learning_rate_runner_policy: float,
                 learning_rate_waiting_policy: float,
                 learning_rate_runner_value: float,
                 learning_rate_waiting_value: float,
                 device: str,
                 **kwargs) -> None:
    reward_sums[agent_identity][current_episode].append(rewards[0])

    q_0 = probs[0]  # .to(device)
    q_1 = next_state_values[0].detach()  # .to(device)
    r = torch.tensor(rewards[0], requires_grad=True, device=device).float()
    bellmann_error = (r + discount_factor * q_1) - q_0
    loss = 0.5 * bellmann_error ** 2

    losses[agent_identity][current_episode].append((loss).squeeze(0).detach().cpu().numpy())

    # Optimize policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if current_episode % 1 == 0:
    #     policy.load_state_dict(value.state_dict())

    if (current_episode * 100 + current_step) % optimizer_message_frequency == 0:
        print('LOSS, REWARD AGENT', str(agent_identity), ': ',
              round(float(loss.detach().cpu().numpy()), 8),
              ', ', float(sum(rewards)))

    del q_0, q_1, r, bellmann_error, loss


def optimize_ACExperienceReplay(net: nn.Module,
                                net2: nn.Module,
                                episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                                optimizer: object,
                                optimizer_value: object,
                                states: List[Tensor],
                                actions: List[Tensor],
                                action_indicees: List[Tensor],
                                log_probs: List[Tensor],
                                probs: List[Tensor],
                                next_states: List[Tensor],
                                rewards: List[Tensor],
                                state_values: List[Tensor],
                                next_state_values: List[Tensor],
                                agent_identity: int,
                                discount_factor: float,
                                reward_sums: List[List[float]],
                                losses: List[List[float]],
                                current_episode: int,
                                current_step: int,
                                optimizer_message_frequency: int,
                                learning_rate_runner_policy: float,
                                learning_rate_waiting_policy: float,
                                learning_rate_runner_value: float,
                                learning_rate_waiting_value: float,
                                device: str,
                                number_samples: int,
                                batch_size: int,
                                runner_identities: List[int],
                                goal_line: int,
                                person_radius: float,
                                **kwargs) -> None:
    states_on_device = torch.cat(
        transform_states(states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs))
    next_states_on_device = torch.cat(
        transform_states(next_states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs))

    all_rewards = torch.tensor(rewards, device=device).float()
    all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    number_steps = int(states_on_device.shape[0] / batch_size + 1)
    net.train()
    for i in range(number_steps):
        start = batch_size * i
        end = batch_size * (i + 1)
        end = min(end, states_on_device.shape[0])
        if start >= end:
            break

        s0 = states_on_device.data[start: end]
        s1 = next_states_on_device.data[start: end]

        # Forward pass
        mean_v, mean_a, log_std_v, log_std_a = net.actor(s0)
        v0 = net.critic(s0)
        v1 = net2.critic(s1)
        v0_old = net2.critic(s0)

        # # Check terminal states and set their values to zero
        # if agent_identity in runner_identities:
        #     for j, state in enumerate(next_states[start: end]):
        #         for id in runner_identities:
        #             if runners_in_goal(state, id, runner_identities, goal_line, person_radius, **kwargs) is True:
        #                 v1[j] = v1[j] * 0.

        # Value Loss
        r = all_rewards[start: end]
        target_v = Variable(r + discount_factor * v1.squeeze(1).detach(), requires_grad=True)
        bellmann_error = v0.squeeze(1) - target_v
        value_loss = 0.5 * torch.mean(bellmann_error ** 2)

        # Policy Loss
        means = torch.cat([mean_v, mean_a], dim=1)
        vars = torch.cat([torch.exp(log_std_v).unsqueeze(0) ** 2 + 1e-4, torch.exp(log_std_a).unsqueeze(0) ** 2 + 1e-4])
        cov = torch.eye(2, device=mean_v.device) * vars
        mn = MultivariateNormal(means, cov)
        action = mn.sample()
        log_probs = mn.log_prob(action).unsqueeze(0)
        # probs = torch.exp(log_probs)
        delta = (v0_old.squeeze(1) - target_v).detach()
        delta.requires_grad = True
        policy_loss = torch.mean(log_probs * delta)

        losses[agent_identity][current_episode].append((policy_loss + value_loss).squeeze(0).detach().cpu().numpy())

        # Optimize policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Optimize value
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    # Update target net
    number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
    if number_of_time_steps_since_training_start % 10 == 0:
        net2.load_state_dict(net.state_dict())

    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float((policy_loss + value_loss).detach().cpu().numpy()), 8))

    del s0, s1, v0, v1, r, states_on_device, next_states_on_device, target_v, bellmann_error, value_loss, policy_loss


def optimize_ACExperienceReplay_KINEMATICS(net: nn.Module,
                                           net2: nn.Module,
                                           episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                                           optimizer: object,
                                           optimizer_value: object,
                                           states: List[Tensor],
                                           actions: List[Tensor],
                                           action_indicees: List[Tensor],
                                           log_probs: List[Tensor],
                                           probs: List[Tensor],
                                           next_states: List[Tensor],
                                           rewards: List[Tensor],
                                           state_values: List[Tensor],
                                           next_state_values: List[Tensor],
                                           agent_identity: int,
                                           discount_factor: float,
                                           reward_sums: List[List[float]],
                                           losses: List[List[float]],
                                           current_episode: int,
                                           current_step: int,
                                           optimizer_message_frequency: int,
                                           learning_rate_runner_policy: float,
                                           learning_rate_waiting_policy: float,
                                           learning_rate_runner_value: float,
                                           learning_rate_waiting_value: float,
                                           device: str,
                                           number_samples: int,
                                           batch_size: int,
                                           runner_identities: List[int],
                                           goal_line: int,
                                           person_radius: float,
                                           **kwargs) -> None:
    states_on_device = torch.cat(states).to(device)
    next_states_on_device = torch.cat(next_states).to(device)

    all_rewards = torch.tensor(rewards, device=device).float()
    all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    number_steps = int(states_on_device.shape[0] / batch_size + 1)
    net.train()
    for i in range(number_steps):
        start = batch_size * i
        end = batch_size * (i + 1)
        end = min(end, states_on_device.shape[0])
        if start >= end:
            break

        s0 = states_on_device.data[start: end]
        s1 = next_states_on_device.data[start: end]

        # Forward pass
        mean_v, mean_a, log_std_v, log_std_a = net.actor(s0)
        v0 = net.critic(s0)
        v1 = net2.critic(s1)
        v0_old = net2.critic(s0)

        # Check terminal states and set their values to zero
        if agent_identity in runner_identities:
            for j, state in enumerate(next_states[start: end]):
                for id in runner_identities:
                    if runners_in_goal(state, id, runner_identities, goal_line, person_radius, **kwargs) is True:
                        v1[j] = v1[j] * 0.

        # Value Loss
        r = all_rewards[start: end]
        target_v = Variable(r + discount_factor * v1.squeeze(1).detach(), requires_grad=True)
        bellmann_error = v0.squeeze(1) - target_v
        value_loss = 0.5 * torch.mean(bellmann_error ** 2)

        # Policy Loss
        means = torch.cat([mean_v, mean_a], dim=1)
        vars = torch.cat([torch.exp(log_std_v).unsqueeze(0) ** 2 + 1e-4, torch.exp(log_std_a).unsqueeze(0) ** 2 + 1e-4])
        cov = torch.eye(2, device=mean_v.device) * vars
        mn = MultivariateNormal(means, cov)
        action = mn.sample()
        log_probs = mn.log_prob(action).unsqueeze(0)
        # probs = torch.exp(log_probs)
        delta = (v0.squeeze(1) - target_v).detach()
        delta.requires_grad = True
        policy_loss = torch.mean(log_probs * delta)

        losses[agent_identity][current_episode].append((policy_loss + value_loss).squeeze(0).detach().cpu().numpy())

        # Optimize policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Optimize value
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    # Update target net
    number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
    if number_of_time_steps_since_training_start % 10 == 0:
        net2.load_state_dict(net.state_dict())

    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float((policy_loss + value_loss).detach().cpu().numpy()), 8))

    del s0, s1, v0, v1, r, states_on_device, next_states_on_device, target_v, bellmann_error, value_loss, policy_loss


def soft_update(actor_net, critic_net, target_actor_net, target_critic_net, TAU):
    for target, src in zip(target_actor_net.parameters(), actor_net.parameters()):
        target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)

    for target, src in zip(target_critic_net.parameters(), critic_net.parameters()):
        target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)


def optimize_DDPG(net: nn.Module,
                  net2: nn.Module,
                  episode_discounted_rewards: Callable[[List[Tensor], float], Tensor],
                  optimizer: object,
                  optimizer_value: object,
                  states: List[Tensor],
                  actions: List[Tensor],
                  action_indicees: List[Tensor],
                  log_probs: List[Tensor],
                  probs: List[Tensor],
                  next_states: List[Tensor],
                  rewards: List[Tensor],
                  state_values: List[Tensor],
                  next_state_values: List[Tensor],
                  agent_identity: int,
                  discount_factor: float,
                  reward_sums: List[List[float]],
                  losses: List[List[float]],
                  current_episode: int,
                  current_step: int,
                  optimizer_message_frequency: int,
                  learning_rate_runner_policy: float,
                  learning_rate_waiting_policy: float,
                  learning_rate_runner_value: float,
                  learning_rate_waiting_value: float,
                  device: str,
                  number_samples: int,
                  batch_size: int,
                  runner_identities: List[int],
                  goal_line: int,
                  person_radius: float,
                  **kwargs) -> None:
    states_on_device = torch.cat(
        transform_states_to_kinematics(states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs))
    next_states_on_device = torch.cat(
        transform_states_to_kinematics(next_states, agent_identity, device, runner_identities, goal_line, person_radius, **kwargs))

    eye = torch.eye(2, device=device).unsqueeze(0)
    number_steps = int(states_on_device.shape[0] / batch_size + 1)
    net.train()
    for i in range(number_steps):
        start = batch_size * i
        end = batch_size * (i + 1)
        end = min(end, states_on_device.shape[0])
        if start >= end:
            break

        s0 = states_on_device.data[start: end]
        s1 = next_states_on_device.data[start: end]

        # Replay
        a = torch.cat(actions[start: end]).to(device)
        q_realized = net.critic([s0, a])
        a1 = net2.actor(s1)
        q1 = net2.critic([s1, a1])

        # Check terminal states and set their values to zero
        if agent_identity in runner_identities:
            for j, state in enumerate(next_states[start: end]):
                for id in runner_identities:
                    if runners_in_goal(state, id, runner_identities, goal_line, person_radius, **kwargs) is True:
                        q1[j] = q1[j] * 0.

        # Value loss
        r = torch.tensor(rewards[start: end], device=device).float()
        target_q = Variable(r + discount_factor * q1.squeeze(1).detach(), requires_grad=True)
        bellmann_error = q_realized.squeeze(1) - target_q
        value_loss = torch.mean(bellmann_error ** 2)

        # Policy Part
        a0 = net.actor(s0)
        q0 = net2.critic([s0, a0])
        policy_loss = -torch.mean(q0)

        losses[agent_identity][current_episode].append((policy_loss + value_loss).squeeze(0).detach().cpu().numpy())

        # Optimize policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Optimize value
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Update target net a little bit
        soft_update(net.actor, net.critic, net2.actor, net2.critic, TAU=0.001)

    number_of_time_steps_since_training_start = sum(kwargs['episode_lengths']) + current_step
    if number_of_time_steps_since_training_start % optimizer_message_frequency == 0:
        print('LOSS AGENT', str(agent_identity), ': ',
              round(float((policy_loss + value_loss).detach().cpu().numpy()), 8))

    del s0, s1, a1, q0, q_realized, q1, r, states_on_device, next_states_on_device, target_q, bellmann_error, value_loss, policy_loss
