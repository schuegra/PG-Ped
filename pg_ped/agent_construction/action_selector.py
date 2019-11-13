from typing import Tuple, Callable, List

import numpy

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import multinomial
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Categorical

from pg_ped.utils import break_if_nan
from pg_ped.visualization.graph_visualization import make_dot


def stochastic_action_selection(state: Tensor, policy: Callable[[Tensor], Tensor], forbidden_actions: List[int],
                                current_episode: int, mode: str, eps_start: float, eps_end: float,
                                eps_decay_length: int) -> Tuple[Tensor]:
    probs = policy.actor(state)
    probs_new = probs.clone()

    # # Introduce noise for exploration (only during training)
    # if mode == 'train':  # Exponential decay of noise epsilon
    #     beta = numpy.log(eps_start)
    #     alpha = (numpy.log(eps_end) - beta) / eps_decay_length
    #     epsilon = numpy.exp(alpha * current_episode + beta)
    #     if current_episode % 100 == 0:
    #         print('EPSILON=', epsilon)
    #
    #     n = MultivariateNormal(loc=torch.zeros_like(probs_new, device=probs_new.device),
    #                            covariance_matrix=epsilon * torch.eye(probs_new.shape[1],
    #                                                                  probs_new.shape[1],
    #                                                                  device=probs_new.device))
    #     noise = n.sample()
    #     probs_new = probs_new + noise  # add noise
    #     probs_new = (probs_new - probs_new.min()) / (probs_new.max() - probs_new.min())  # normalize
    #     probs_new = probs_new / probs_new.sum()  # set sum to 1
    #     del noise, n, alpha, beta, epsilon

    probs_new[0, forbidden_actions] = 0
    if bool(torch.sum(probs_new) < 1e-8) is True:
        probs_new = probs_new + torch.ones_like(probs_new)
        probs_new[0, forbidden_actions] = 0
    probs_new = probs_new / probs_new.sum()

    # print(torch.mean(probs - probs_new))
    if bool(torch.any(probs_new < 0)) is True:
        print(probs_new)
        print('negative probs! -> abort episode')
        return torch.tensor(-1, device=state.device), \
               torch.tensor(-5, device=state.device), \
               torch.tensor(0, device=state.device), \
               True
    c = Categorical(probs_new)

    failed = False
    action_index = c.sample()
    try:
        action_index.cpu()  # on the gpu, the exception will not be detected
    except Exception as e:
        print('ERROR MESSAGE: ', e)
        return torch.tensor(-1, device=state.device), \
               torch.tensor(-5, device=state.device), \
               torch.tensor(0, device=state.device), \
               True

    del probs_new

    return action_index, c.log_prob(action_index).unsqueeze(0), probs[0, action_index], failed


def continous_action_selection_normal(state: Tensor, policy: Callable[[Tensor], Tensor],
                                      fobidden_actions: List[int], mode: str,
                                      current_episode, *dummies) -> Tuple[Tensor]:
    #policy.eval()
    mean_v, mean_a, log_std_v, log_std_a = policy.actor(state)
    #action = torch.cat([mean_v, mean_a]).transpose(1, 0)
    #prob, log_prob = torch.ones(1), torch.ones(1)
    mean = torch.cat([mean_v, mean_a]).transpose(1, 0)
    #stds = torch.cat([torch.exp(log_std_v).unsqueeze(0), torch.exp(log_std_a).unsqueeze(0)])
    if mode == 'simulate':
        stds = 0.01 * torch.ones_like(mean)
    else:
        #stds = stds * (1 - current_episode * 1. / 1e4)
        #stds = torch.max(0.01 * torch.ones_like(stds), stds)
        stds = 0.5 * torch.ones_like(mean)
    vars = stds ** 2
    cov = torch.eye(2, device=mean_v.device) * vars
    mn = MultivariateNormal(mean, cov)
    action = mn.sample()
    # action[1:] += (torch.rand(1, device=action.device) - 0.5) # + noise for better exploration
    #log_prob = (mn.log_prob(action) - mn.log_prob(mean)).unsqueeze(0)
    log_prob = mn.log_prob(action).unsqueeze(0)
    prob = torch.exp(log_prob)
    from pg_ped.utils import break_if_nan
    break_if_nan(action)

    return action, log_prob, prob, False


def ddpg_action_computation(state: Tensor, policy: Callable[[Tensor], Tensor],
                            fobidden_actions: List[int], *dummies) -> Tuple[Tensor]:
    #policy.eval()
    delta_v, delta_a = policy(state)
    action = torch.cat([delta_v, delta_a])
    # action = action + 0.5*(torch.rand(action.shape, device=action.device) - 0.5)
    return action, torch.zeros(1), torch.zeros(1), False


def mixed_action_selection_normal(state: Tensor, policy: Callable[[Tensor], Tensor]) -> Tuple[Tensor]:
    net_output = policy(state)
    mean_std = net_output[0, :2]
    n = Normal(mean_std[0], mean_std[1])
    angle = n.sample().unsqueeze(0)
    log_prob_angle = n.log_prob(angle)
    c = Categorical(net_output[0, 2:])
    step_length_index = c.sample().unsqueeze(0)
    log_prob_step_length_index = c.log_prob(step_length_index)
    log_prob = log_prob_step_length_index + log_prob_angle
    action = torch.cat([angle, step_length_index.float()])
    return action.unsqueeze(0), log_prob


def epsilon_greedy(state: Tensor, policy: Callable[[Tensor], Tensor],
                   forbidden_actions: List[int], current_overall_step: int, mode: str,
                   eps_start: float, eps_end: float, eps_decay_length: int,
                   start_learning: int) -> Tuple[Tensor]:
    with torch.no_grad():
        #policy.eval()
        scores = policy(state)
    scores_new = scores.clone()
    del scores
    scores_new[0, forbidden_actions] = -1e10  # avoid these actions

    failed = False

    number_actions = scores_new.shape[1]
    max_action = torch.argmax(scores_new)
    random_number = torch.rand(1)


    if mode in ['train', 'resumetraining']:  # Exponential decay of noise epsilon
        #beta = numpy.log(eps_start)
        #alpha = (numpy.log(eps_end) - beta) / eps_decay_length
        #epsilon = numpy.exp(alpha * current_overall_step + beta)
        epsilon = eps_start - (current_overall_step - start_learning) * (eps_start - eps_end) / eps_decay_length
        epsilon = max(min(eps_start, epsilon), eps_end) #
        # if current_episode % 100 == 0:
        #     print('EPSILON=', epsilon)
    elif mode == 'simulate':
        epsilon = 0.00

    if bool(random_number < 1 - epsilon) is True:
        action_index = max_action.unsqueeze(0)
    else:
        action_index = torch.randint(0, number_actions, (1,))

    try:
        action_index.cpu()  # on the gpu, the exception will not be detected
    except Exception as e:
        print('ERROR MESSAGE: ', e)
        return torch.tensor(-1, device=state.device), \
               torch.tensor(-5, device=state.device), \
               torch.tensor(0, device=state.device), \
               True

    return action_index, torch.log(scores_new[0, action_index]), scores_new[0, action_index], failed


def thompson_sampling(state: Tensor, policy: Callable[[Tensor], Tensor],
                      forbidden_actions: List[int], current_episode: int, mode: str,
                      eps_start: float, eps_end: float, eps_decay_length: int) -> Tuple[Tensor]:
    scores = policy(state)
    scores_new = scores.clone()
    del scores
    scores_new[0, forbidden_actions] = -1e10  # avoid these actions

    failed = False

    number_actions = scores_new.shape[1]
    action_index = torch.argmax(scores_new).unsqueeze(0)

    try:
        action_index.cpu()  # on the gpu, the exception will not be detected
    except Exception as e:
        print('ERROR MESSAGE: ', e)
        return torch.tensor(-1, device=state.device), \
               torch.tensor(-5, device=state.device), \
               torch.tensor(0, device=state.device), \
               True

    return action_index, torch.log(scores_new[0, action_index]), scores_new[0, action_index], failed
