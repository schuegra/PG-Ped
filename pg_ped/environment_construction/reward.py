import numpy
from scipy.stats import truncnorm
from torch import Tensor

from pg_ped import config
from pg_ped.environment_construction.termination import *
from pg_ped.utils import *
from pg_ped.environment_construction.geometry import *

EPS = numpy.finfo('float').eps


def reward_gravity(state: Tensor,
                   agent_identity: int,
                   influence_radius: float,
                   standard_deviation: float,
                   goal_line: float,
                   person_radius: float,
                   runner_identities: int,
                   device: str,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   soft_person_radius: float,
                   initial_state: Tensor,
                   **kwargs) -> Tensor:
    """
        Reward function as in martinez-gil et al. (year).

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    if agent_identity in runner_identities:
        current_position = state[agent_identity, :2].detach().cpu().numpy()
        dist_squared = (goal_line - current_position[1]) ** 2
        scalar_reward += 1. / (1 + dist_squared)

    else:
        # initial_position = initial_state[agent_identity, :2].detach().cpu().numpy()
        # current_position = state[agent_identity, :2].detach().cpu().numpy()
        # dist_squared = (initial_position[0] - current_position[0]) ** 2 + (initial_position[1] - current_position[1]) ** 2
        # scalar_reward += 1./(1+dist_squared)
        if check_movement(state, agent_identity, variables_per_agent_per_timestep, backward_view):
            scalar_reward -= 1

    scalar_reward -= density(state, agent_identity, influence_radius, standard_deviation)

    return scalar_reward


def reward_personal_space(state: Tensor,
                          agent_identity: int,
                          influence_radius: float,
                          standard_deviation: float,
                          goal_line: float,
                          person_radius: float,
                          runner_identity: int,
                          device: str,
                          variables_per_agent_per_timestep: int,
                          backward_view: int,
                          soft_person_radius: float,
                          **kwargs) -> Tensor:
    scalar_reward = 0

    if agent_identity == runner_identity:
        if agent_in_goal(state,
                         agent_identity=runner_identity,
                         goal_line=goal_line,
                         runner_identity=runner_identity,
                         person_radius=person_radius) is False:
            scalar_reward -= 1
        else:
            scalar_reward += 40

        if check_boundary_collision(state, runner_identity, person_radius=person_radius, **kwargs) is True:
            scalar_reward -= 15
        #
        # if check_personal_space(state, agent_identity, soft_person_radius, EPS,
        #                         variables_per_agent_per_timestep)[0] is True:
        #     scalar_reward -= 4

    else:
        # # Punish waiting when passing goal line
        # if agent_in_goal(state,
        #                  agent_identity=agent_identity,
        #                  goal_line=goal_line,
        #                  runner_identity=runner_identity,
        #                  person_radius=person_radius) or \
        if check_boundary_collision(state, agent_identity, person_radius=person_radius, **kwargs) is True:
            scalar_reward -= 15

        if check_personal_space(state, agent_identity, soft_person_radius, EPS,
                                variables_per_agent_per_timestep)[0] is True:
            scalar_reward -= 4

        if check_movement(state, agent_identity, variables_per_agent_per_timestep, backward_view):
            scalar_reward -= 1

    return scalar_reward


def reward(state: Tensor,
           agent_identity: int,
           influence_radius: float,
           standard_deviation: float,
           goal_line: float,
           person_radius: float,
           soft_person_radius: float,
           runner_identities: List[int],
           device: str,
           initial_state: Tensor,
           variables_per_agent_per_timestep: int,
           backward_view: int,
           start_line: float,
           y_min: float,
           y_max: float,
           **kwargs) -> Tensor:
    """
        Reward function.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    p = state[agent_identity, :2]
    p_old = state[agent_identity, 4:6]

    if agent_identity in runner_identities:

        # Goal reward
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) is True:
            scalar_reward += 0.  # 100.
        else:
            scalar_reward -= 1.  # 0.

        # # Wall collision
        if check_boundary_collision(state, agent_identity, person_radius=person_radius,
                                    y_min=start_line, y_max=y_max + 1., **kwargs) is True:
            scalar_reward -= 10.

        # Agent collision
        if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
            scalar_reward -= 5.

    else:

        # # Stay in box
        if check_boundary_collision(state, agent_identity, person_radius=person_radius,
                                    y_min=y_min, y_max=y_max, **kwargs) is True:
            scalar_reward -= 10.

        if check_movement(state, agent_identity, variables_per_agent_per_timestep, backward_view, tol=0.05) is True:
            scalar_reward -= 1.

        # Agent collision
        if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
            scalar_reward -= 5.

    return scalar_reward


def reward2(state: Tensor,
            agent_identity: int,
            influence_radius: float,
            standard_deviation: float,
            goal_line: float,
            person_radius: float,
            soft_person_radius: float,
            runner_identities: List[int],
            device: str,
            initial_state: Tensor,
            variables_per_agent_per_timestep: int,
            backward_view: int,
            start_line: float,
            y_min: float,
            y_max: float,
            **kwargs) -> Tensor:
    """
        Reward function.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    p = state[agent_identity, :2]
    p_old = state[agent_identity, 4:6]

    if agent_identity in runner_identities:

        # Goal reward
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) is True:
            scalar_reward += 100.  # 30.
        else:
            scalar_reward -= 0.  # 1.

        # # Wall collision
        if check_boundary_collision(state, agent_identity, person_radius=soft_person_radius,
                                    y_min=start_line, y_max=y_max + 1., **kwargs) is True:
            scalar_reward -= 5.

        goal = torch.tensor([state[agent_identity, 0], goal_line], device=device)

        # # Agent collision
        # if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
        #     scalar_reward -= 2.

        # # Moving with small angle to goal
        angle = angle_2D_full(p - p_old, goal - state[agent_identity, :2])
        if bool(angle < 0.1) or bool(angle > 2 * math_pi - 0.1):
            scalar_reward += 0.1

        # scalar_reward = 2 * ((scalar_reward + 1.)/(100. + 1.) - 0.5)


    else:
        # Goal reward
        # goal = initial_state[agent_identity, :2]
        # if bool(torch.norm(goal - p) < person_radius):
        #     scalar_reward += 1.

        # # Wall collision
        # if check_boundary_collision(state, agent_identity, person_radius=soft_person_radius,
        #                             y_min=start_line, y_max=y_max, **kwargs):
        #     scalar_reward -= 1.

        # # Stay in box
        if check_boundary_collision(state, agent_identity, person_radius=soft_person_radius,
                                    y_min=y_min, y_max=y_max, **kwargs) is True:
            scalar_reward -= 5.

        if check_movement(state, agent_identity, variables_per_agent_per_timestep, backward_view, tol=0.05) is False:
            scalar_reward += 1.

        # Agent collision
        if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
            scalar_reward -= 5.

        # # Moving with small angle to goal
        # angle = angle_2D_full(p - p_old, goal - state[agent_identity, :2])
        # if bool(angle < 0.1) or bool(angle > 2 * math_pi - 0.1):
        #     scalar_reward += 0.1

        # scalar_reward = 2 * ((scalar_reward + 2.)/(2. + 1.) - 0.5)

    return scalar_reward


def reward3(state: Tensor,
            agent_identity: int,
            influence_radius: float,
            standard_deviation: float,
            goal_line: float,
            person_radius: float,
            soft_person_radius: float,
            runner_identities: List[int],
            device: str,
            initial_state: Tensor,
            variables_per_agent_per_timestep: int,
            backward_view: int,
            start_line: float,
            y_min: float,
            y_max: float,
            **kwargs) -> Tensor:
    """
        Reward function.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    p = state[agent_identity, :2]
    p_old = state[agent_identity, 4:6]

    if agent_identity in runner_identities:

        # Goal reward
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) is True:
            scalar_reward += 100.
        else:
            scalar_reward -= 1.  # 0.

        # # Wall collision
        # if check_boundary_collision(state, agent_identity, person_radius=person_radius,
        #                             y_min=start_line, y_max=y_max + 10., **kwargs) is True:
        #     scalar_reward -= 10.
        #
        # # Agent collision
        # if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
        #     scalar_reward -= 5.

    else:

        # Wall collision
        if check_boundary_collision(state, agent_identity, person_radius=person_radius,
                                    y_min=start_line, y_max=y_max, **kwargs) is True:
            scalar_reward -= 10.

        # Inside Box
        if check_boundary_collision(state, agent_identity, y_min=y_min, y_max=goal_line,
                                    person_radius=person_radius, **kwargs) is True:
            scalar_reward -= 8.

        # # Goal: second last position
        # goal = state[agent_identity, 2 * 4: 2 * 4 + 2]
        # distance_goal = torch.norm(state[agent_identity, :2] - goal)
        # distance_goal = distance_goal.cpu().numpy()
        # T = 0.3
        # n = truncnorm(-T, T)
        # scalar_reward += n.pdf(distance_goal)

        # Agent collision
        if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
            scalar_reward -= 5.

    return scalar_reward


def reward4(state: Tensor,
            agent_identity: int,
            influence_radius: float,
            standard_deviation: float,
            goal_line: float,
            person_radius: float,
            soft_person_radius: float,
            runner_identities: List[int],
            device: str,
            initial_state: Tensor,
            variables_per_agent_per_timestep: int,
            backward_view: int,
            start_line: float,
            y_min: float,
            y_max: float,
            **kwargs) -> Tensor:
    """
        Reward function.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    p = state[agent_identity, :2]
    p_old = state[agent_identity, 4:6]

    if agent_identity in runner_identities:
        scalar_reward += 0. # does not learn
    else:
        # targetPoint = torch.tensor([0.5, 8.0], device=device)
        # dist = torch.norm(targetPoint - p).cpu().numpy()
        # if dist < 0.8:
        #     scalar_reward += 10.
        #scalar_reward -= dist

        # Runner gets to goal
        # if agent_in_goal(state,
        #                  agent_identity=runner_identities[0],
        #                  goal_line=goal_line,
        #                  runner_identities=runner_identities,
        #                  person_radius=person_radius) is True:
        #     scalar_reward += 100.
        # else:
        #     scalar_reward -= 0.

        # Maximise distance to other waiting (2 waiting)
        pers_id_list = list(config.cli.pers.getIDList())
        runner_id = pers_id_list[-1]
        pers_id_list = pers_id_list[:-1]
        my_id = pers_id_list[agent_identity - 1]
        my_pos = config.cli.pers.getPosition2D(my_id)
        pers_id_list.remove(my_id)
        other_id = pers_id_list[0]
        other_pos = config.cli.pers.getPosition2D(other_id)
        sos = (other_pos[0] - my_pos[0]) ** 2 + (other_pos[1] - my_pos[1]) ** 2
        dist = sos ** 0.5
        if dist >= 1.:
            scalar_reward += 1.

        # # Try to get to target
        # pers_id_list = list(config.cli.pers.getIDList())
        # runner_id = pers_id_list[-1]
        # pers_id_list = pers_id_list[:-1]
        # my_id = pers_id_list[agent_identity - 1]
        # my_pos = config.cli.pers.getPosition2D(my_id)
        # my_target_id = config.cli.pers.getTargetList(my_id)[0]
        # my_target_dist = config.cli.poly.getDistance(my_target_id, my_pos[0], my_pos[1])
        # scalar_reward -= my_target_dist

        # # Choose closer target
        # pers_id_list = list(config.cli.pers.getIDList())
        # runner_id = pers_id_list[-1]
        # pers_id_list = pers_id_list[:-1]
        # my_id = pers_id_list[agent_identity - 1]
        # my_pos = config.cli.pers.getPosition2D(my_id)
        # if my_pos[0] > 1.5:
        #     my_target_id = config.cli.pers.getTargetList(my_id)[0]
        #     if my_target_id == "8":
        #         scalar_reward += 1.
        # else:
        #     my_target_id = config.cli.pers.getTargetList(my_id)[0]
        #     if my_target_id == "7":
        #         scalar_reward += 1.

    return scalar_reward


def reward_simplified(state: Tensor,
                      agent_identity: int,
                      influence_radius: float,
                      standard_deviation: float,
                      goal_line: float,
                      person_radius: float,
                      soft_person_radius: float,
                      runner_identities: List[int],
                      device: str,
                      initial_state: Tensor,
                      variables_per_agent_per_timestep: int,
                      backward_view: int,
                      start_line: float,
                      x_min: float,
                      x_max: float,
                      y_min: float,
                      y_max: float,
                      **kwargs) -> Tensor:
    """
        Reward function.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    p = state[agent_identity, :2]
    p_old = state[agent_identity, 4:6]

    if agent_identity in runner_identities:

        # Goal reward
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) is True:
            scalar_reward += 100.
        # else:
        #     scalar_reward -= 1.

        # Encourage walking straight forward
        goal = torch.tensor([state[agent_identity, 0], goal_line], device=state.device) - state[agent_identity, :2]
        shift = state[agent_identity, :2] - state[agent_identity, 4:6]
        angle = angle_2D_full(goal, shift)
        if bool(angle < math_pi / 8.) or bool(angle > 2 * math_pi - math_pi / 8.):
            scalar_reward += 0.1

        # orientation = new_state[agent_identity, 2:4]
        # backward_movement = torch.dot(orientation, shift) < 0
        # if bool(backward_movement) is True:
        #
    #
    # else:

    # # Inside Box
    # if check_boundary_collision(state, agent_identity, x_min=x_min, x_max=x_max, y_min=y_min, y_max=goal_line,
    #                             person_radius=person_radius, **kwargs) is False:
    #     scalar_reward += 1.

    # Running above each other
    # scalar_reward -= accumulated_density(state[agent_identity, 4:6], state[agent_identity, :2],
    #                                      state[:, :2], agent_identity,
    #                                      influence_radius, standard_deviation).cpu().numpy()

    # Running out of bounds
    if check_boundary_collision(state, agent_identity, x_min, x_max, start_line, y_max, person_radius,
                                **kwargs) is True:
        scalar_reward -= 10.

    col_id, _ = collision_during_step2(state, agent_identity, person_radius, EPS)
    if col_id != -1:
        scalar_reward -= 10.

    # Check agent collision
    if check_agent_collision(state, agent_identity, person_radius, EPS) is True:
        scalar_reward -= 1.

    if check_agent_collision(state, agent_identity, soft_person_radius, EPS) is True:
        scalar_reward -= .5

    return scalar_reward


def reward_accumulated_density(state: Tensor,
                               agent_identity: int,
                               influence_radius: float,
                               standard_deviation: float,
                               goal_line: float,
                               person_radius: float,
                               soft_person_radius: float,
                               runner_identities: List[int],
                               device: str,
                               initial_state: Tensor,
                               variables_per_agent_per_timestep: int,
                               backward_view: int,
                               start_line: float,
                               x_min: float,
                               x_max: float,
                               y_min: float,
                               y_max: float,
                               **kwargs) -> Tensor:
    """
        Reward function.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    p = state[agent_identity, :2]
    p_old = state[agent_identity, 4:6]

    if agent_identity in runner_identities:

        # Goal reward
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) is True:
            scalar_reward += 100.
        else:
            scalar_reward -= 1.

        # # Encourage walking straight forward
        # goal = torch.tensor([state[agent_identity, 0], goal_line], device=state.device) - state[agent_identity, :2]
        # shift = state[agent_identity, :2] - state[agent_identity, 4:6]
        # angle = angle_2D_full(goal, shift)
        # if bool(angle < math_pi / 8.) or bool(angle > 2 * math_pi - math_pi / 8.):
        #     scalar_reward += 0.1

        # orientation = new_state[agent_identity, 2:4]
        # backward_movement = torch.dot(orientation, shift) < 0
        # if bool(backward_movement) is True:
        #
    #
    else:

        # Inside Box
        if check_boundary_collision(state, agent_identity, x_min=x_min, x_max=x_max, y_min=y_min, y_max=goal_line,
                                    person_radius=person_radius, **kwargs) is False:
            scalar_reward += 1.

    # Running out of bounds
    if check_boundary_collision(state, agent_identity, x_min, x_max, start_line, y_max, person_radius,
                                **kwargs) is True:
        scalar_reward -= 10.
    #
    # # Accumulated density
    # start = state[agent_identity, 4:6]
    # goal = state[agent_identity, :2]
    # scalar_reward -= float(accumulated_density(start, goal, state[:, :2].clone(), agent_identity,
    #                                            influence_radius, standard_deviation).cpu().numpy())

    return scalar_reward


def reward_martinez_gil(state: Tensor,
                        agent_identity: int,
                        influence_radius: float,
                        standard_deviation: float,
                        goal_line: float,
                        person_radius: float,
                        runner_identities: int,
                        device: str,
                        variables_per_agent_per_timestep: int,
                        backward_view: int,
                        soft_person_radius: float,
                        **kwargs) -> Tensor:
    """

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = 0

    if agent_identity in runner_identities:
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) is True:
            scalar_reward += 10.
        else:
            scalar_reward -= 1.
        # scalar_reward -= float(((state[agent_identity, 1] - goal_line) ** 2).detach().cpu().numpy())

        # if check_movement(state, agent_identity, variables_per_agent_per_timestep, backward_view) is False:
        #     scalar_reward -= 1.

        if check_boundary_collision(state, agent_identity, person_radius=soft_person_radius, **kwargs):
            scalar_reward -= 1.
        #
        if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
            scalar_reward -= 1.

        # r := (r - worst_possible)/new_highest_possible
        # scalar_reward = (scalar_reward + 20)/70.
        # scalar_reward -= density(state[:, :2], agent_identity, influence_radius, standard_deviation)

        scalar_reward = 2 * (((scalar_reward + 3.) / 13.) - 0.5)  # scale to [-1,1]

    else:
        # # Punish waiting when passing goal line
        if agent_in_goal(state,
                         agent_identity=agent_identity,
                         goal_line=goal_line,
                         runner_identities=runner_identities,
                         person_radius=person_radius) or \
                check_boundary_collision(state, agent_identity, person_radius=soft_person_radius, **kwargs):
            scalar_reward -= 2.
        #
        if check_agent_collision(state, agent_identity, soft_person_radius, EPS)[0] is True:
            scalar_reward -= 1.
        # # scalar_reward -= density(state, agent_identity, influence_radius, standard_deviation)
        #

        if check_movement(state, agent_identity, variables_per_agent_per_timestep, backward_view, tol=0.05) is True:
            scalar_reward -= .5

        # scalar_reward = (scalar_reward + 20)/20.
        # scalar_reward -= density(state[:, :2], agent_identity, influence_radius, standard_deviation)

        scalar_reward = 2 * (((scalar_reward + 3.5) / 3.5) - 0.5)  # scale to [-1,1]

    return scalar_reward


def reward_teamwork(state: Tensor,
                    agent_identity: int,
                    influence_radius: float,
                    standard_deviation: float,
                    goal_line: float,
                    person_radius: float,
                    runner_identities: List[int],
                    device: str,
                    **kwargs) -> Tensor:
    """
        Reward function as in martinez-gil et al. (year).

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    for runner in runner_identities:
        scalar_reward = - float(((state[runner, 1] - goal_line) ** 2).detach().cpu().numpy())

    # if agent_in_goal(state,
    #                  agent_identity=runner_identity,
    #                  goal_line=goal_line,
    #                  runner_identity=runner_identity,
    #                  person_radius=person_radius) is False:
    #     scalar_reward -= 1
    # else:
    #     scalar_reward += 40

    # if check_boundary_collision(state, runner_identity, person_radius=person_radius, **kwargs):
    #     scalar_reward -= 15
    #
    # if check_agent_collision(state, agent_identity, person_radius, EPS) is True:
    #     scalar_reward -= 4

    return scalar_reward

    return torch.FloatTensor([scalar_reward]).to(device)


def reward_go_back(state: Tensor,
                   agent_identity: int,
                   influence_radius: float,
                   standard_deviation: float,
                   goal_line: float,
                   person_radius: float,
                   runner_identity: int,
                   device: str,
                   initial_state: Tensor,
                   **kwargs) -> Tensor:
    """
        Reward function as in martinez-gil et al. (year) extended
        by enccouraging movement back to the initial position of the waiting.

        Parameters:
        -----------

        state: Tensor
            A tensor of 2D cartesian coordinates.

    """

    scalar_reward = -density(state[:, :2], agent_identity, influence_radius, standard_deviation)

    if agent_identity == runner_identity:
        if agent_in_goal(state,
                         agent_identity=runner_identity,
                         goal_line=goal_line,
                         runner_identity=runner_identity,
                         person_radius=person_radius) is True:
            scalar_reward += 100

        # if check_boundary_collision(state, runner_identity, person_radius=person_radius, **kwargs):
        #     scalar_reward -= 15
        #
        # if check_agent_collision(state, agent_identity, person_radius) is True:
        #     scalar_reward -= 4

    else:
        # # Punish waiting when passing goal line
        # if agent_in_goal(state,
        #                  agent_identity=agent_identity,
        #                  goal_line=goal_line,
        #                  runner_identity=runner_identity,
        #                  person_radius=person_radius) or \
        #         check_boundary_collision(state, agent_identity, person_radius=person_radius, **kwargs):
        #     scalar_reward -= 15

        # if check_agent_collision(state, agent_identity, person_radius) is True:
        #     scalar_reward -= 4

        initial_position = initial_state[agent_identity, :2]
        position = state[agent_identity, :2]
        distance_to_initial_position = torch.norm(initial_position - position, p=2)
        scalar_reward -= distance_to_initial_position

    return scalar_reward
