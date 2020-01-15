from typing import List

import torch
from torch import Tensor

from .geometry import check_agent_passed_goal_line


def max_steps_termination(state: Tensor, current_step=0, max_steps=100, **kwargs) -> bool:
    if current_step < max_steps:
        return False
    else:
        return True


def random_termination(state: Tensor, **kwargs) -> bool:
    termination_prob = 0.001
    sample = torch.rand(1)
    return bool((sample < termination_prob)[0])


def agent_in_goal(state: Tensor,
                  agent_identity: int,
                  runner_identities: List[int],
                  goal_line: float,
                  person_radius: float,
                  device: str,
                  **kwargs) -> bool:
    if agent_identity in runner_identities:
        return check_agent_passed_goal_line(state, agent_identity, goal_line, person_radius, device)
    else:
        return False


def runners_in_goal(state: Tensor,
                    agent_identity: int,
                    runner_identities: List[int],
                    goal_line: float,
                    person_radius: float,
                    device: str,
                    **kwargs) -> bool:
    terminate = True
    for runner_identity in runner_identities:
        runner_passed_goal_line = check_agent_passed_goal_line(state, runner_identity, goal_line, person_radius, device)
        terminate = terminate and runner_passed_goal_line
    return terminate
