from typing import Tuple, Callable

import numpy

import torch
from torch import Tensor

from pg_ped import config
from pg_ped.environment_construction.geometry import *

from pg_ped.utils import break_if_nan

EPS = numpy.finfo(dtype=numpy.float).eps


def choose_target(state: Tensor,
                  agent_identity: int,
                  action: Tensor,
                  person_radius: float,
                  soft_person_radius: float,
                  time_per_step: float,
                  variables_per_agent_per_timestep: int,
                  backward_view: int,
                  x_min: float,
                  x_max: float,
                  y_min: float,
                  y_max: float,
                  goal_line: float,
                  step_length_runner: float,
                  runner_identities: List[int],
                  push_dist: float,
                  maximum_speed_forward: float,
                  maximum_speed_backward: float,
                  start_line: float,
                  **kwargs) -> Tensor:
    '''
    '''

    try:
        # Set target
        if agent_identity > 0:

            person_id = str(agent_identity)
            poly_ids = config.cli.poly.getIDList()
            target_ids = []
            for id in poly_ids:
                targetType = config.cli.poly.getType(id)
                if targetType == "TARGET":
                    target_ids += [id]
            target_ids = [x for x in target_ids if x != "6"] # "6" is assumed to be the target of the runner
            target_index = action[0].cpu().numpy()
            target_id = str(target_ids[target_index])
            config.cli.pers.setNextTargetListIndex(person_id, 0)
            config.cli.pers.setTargetList(person_id, [target_id])

        elif agent_identity == 0:
            person_id = max(config.cli.pers.getIDList())
            config.cli.pers.setNextTargetListIndex(person_id, 0)

        # Update position
        new_state = state.clone()
        positions_dict = config.cli.pers.getPosition2DList()
        positions_in_order_of_vadere = list(positions_dict.values())
        position_runner = [positions_in_order_of_vadere[-1]]
        positions_other = [pos for pos in positions_in_order_of_vadere[:-1]]
        positions = position_runner + positions_other
        new_state = update_state_agent(new_state, agent_identity, positions[agent_identity], variables_per_agent_per_timestep, backward_view, **kwargs)


    except Exception as e:
        print('ERROR MESAGE in state_transition/choose_target: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False


def push_move_heuristic(state: Tensor,
                        agent_identity: int,
                        variables_per_agent_per_timestep: int,
                        backward_view: int,
                        goal_x: float,
                        goal_y: float,
                        goal_line: float,
                        x_min: float,
                        x_max: float,
                        y_min: float,
                        y_max: float,
                        person_radius: float,
                        step_length: float,
                        push_dist: float,
                        push_func: Callable[[object], object]) -> Tensor:
    '''
    Uses pushing as a heuristic in movement
    '''
    failed = False

    col_id, dist = collision_during_step(state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)

    new_state = state.clone()
    del state
    x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
    dx, dy = goal_x - x0, goal_y - y0
    goal_dist = torch.sqrt(dx ** 2 + dy ** 2)
    dx, dy = dx * step_length / (goal_dist + EPS), dy * step_length / (goal_dist + EPS)
    v = torch.sqrt(dx ** 2 + dy ** 2)
    m = dy / (dx + EPS)
    t = y0 - m * x0

    if col_id >= 0:
        xi0, yi0 = new_state[col_id, 0], new_state[col_id, 1]
        x1, y1 = x0 + dist * dx / (v + EPS), y0 + dist * dy / (v + EPS)
        x1, y1 = stop_at_boundaries(x0, y0, x1, y1, x_min, x_max, y_min, y_max, person_radius, EPS)
        xi1, yi1 = push_func(xi0, yi0, x1, y1, dx, dy, t, v, push_dist, EPS,
                             variables_per_agent_per_timestep, backward_view)
        xi1, yi1 = stop_at_boundaries(xi0, yi0, xi1, yi1, x_min, x_max, y_min, y_max, person_radius, EPS)
        try:
            new_state, failed = push_heuristic(new_state, col_id, variables_per_agent_per_timestep, backward_view, xi1,
                                               yi1, goal_line, x_min, x_max, y_min, y_max, person_radius, push_dist,
                                               push_dist, push_func)
        except Exception as e:
            print("Error message: ", e)
            failed = True
            return new_state, failed

    else:
        failed = True
        return new_state, failed

    if failed is False:
        new_state, failed = update_state(new_state, x1, y1, agent_identity, variables_per_agent_per_timestep,
                                         backward_view, person_radius, x_min, x_max, y_min, y_max, EPS)
    failed = break_if_nan(new_state) or failed
    return new_state, failed


def push_heuristic(state: Tensor,
                   agent_identity: int,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   goal_x: float,
                   goal_y: float,
                   goal_line: float,
                   x_min: float,
                   x_max: float,
                   y_min: float,
                   y_max: float,
                   person_radius: float,
                   step_length: float,
                   push_dist: float,
                   push_func: Callable[[object], object]) -> Tensor:
    '''
    Manages the chain reaction of collision by recursively trying to move to a point, if no collision happens.

    :param state:
    :param agent_identity:
    :param variables_per_agent_per_timestep:
    :param backward_view:
    :param goal_x:
    :param goal_y:
    :param goal_line:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param person_radius:
    :param step_length:
    :param push_dist:
    :param push_func:
    :return:
    '''
    failed = False

    col_id, dist = collision_during_step(state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)

    new_state = state.clone()
    del state
    x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
    dx, dy = goal_x - x0, goal_y - y0
    goal_dist = torch.sqrt(dx ** 2 + dy ** 2)
    dx, dy = dx * step_length / (goal_dist + EPS), dy * step_length / (goal_dist + EPS)
    v = torch.sqrt(dx ** 2 + dy ** 2)
    m = dy / (dx + EPS)
    t = y0 - m * x0

    if col_id >= 0:
        xi0, yi0 = new_state[col_id, 0], new_state[col_id, 1]
        x1, y1 = x0 + dist * dx / (v + EPS), y0 + dist * dy / (v + EPS)
        x1, y1 = stop_at_boundaries(x0, y0, x1, y1, x_min, x_max, y_min, y_max, person_radius, EPS)
        xi1, yi1 = push_func(xi0, yi0, x1, y1, dx, dy, t, v, push_dist, EPS,
                             variables_per_agent_per_timestep, backward_view)
        xi1, yi1 = stop_at_boundaries(xi0, yi0, xi1, yi1, x_min, x_max, y_min, y_max, person_radius, EPS)
        try:
            new_state, failed = push_heuristic(new_state, col_id, variables_per_agent_per_timestep, backward_view, xi1,
                                               yi1, goal_line, x_min, x_max, y_min, y_max, person_radius, push_dist,
                                               push_dist, push_func)
        except Exception as e:
            print("Error message: ", e)
            failed = True
            return new_state, failed

    else:  # Move directly towards goal
        x1, y1 = x0 + dx, y0 + dy
        x1, y1 = stop_at_boundaries(x0, y0, x1, y1, x_min, x_max, y_min, y_max, person_radius, EPS)

    if failed is False:
        new_state, failed = update_state(new_state, x1, y1, agent_identity, variables_per_agent_per_timestep,
                                         backward_view, person_radius, x_min, x_max, y_min, y_max, EPS)
        if failed is True:
            print('FAILED UPDATING STATE')
    failed = break_if_nan(new_state) or failed
    if break_if_nan(new_state) is True:
        print('NANS')
    return new_state, failed


# def two_ways_evasion_heuristic(state: Tensor,
#                                agent_identity: int,
#                                evasion_func: Callable[[Tuple[Tensor]], Tuple[Tensor]],
#                                which_side: str,
#                                variables_per_agent_per_timestep: int,
#                                backward_view: int,
#                                goal_x: float,
#                                goal_y: float,
#                                goal_line: float,
#                                x_min: float,
#                                x_max: float,
#                                y_min: float,
#                                y_max: float,
#                                person_radius: float,
#                                step_length: float,
#                                push: bool,
#                                push_dist: float,
#                                push_func: Callable[[object], object]) -> Tensor:
#     col_id, _ = collision_during_step(state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)
#     p0 = state[agent_identity, :2].clone()
#     new_state = state.clone()
#     del state
#     failed = False
#
#     if col_id >= 0:  # Evade
#
#         pi = new_state[col_id, :2]
#         x11, y11, x12, y12 = evasion_func(pi, p0, person_radius, step_length, EPS)
#         x11, y11 = stop_at_boundaries(p0[0], p0[1], x11, y11, x_min, x_max, y_min, y_max, person_radius, EPS)
#         x12, y12 = stop_at_boundaries(p0[0], p0[1], x12, y12, x_min, x_max, y_min, y_max, person_radius, EPS)
#         if which_side is 'left':
#             step_length = torch.sqrt((x11 - p0[0]) ** 2 + (y11 - p0[1]) ** 2)
#             new_state, failed = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
#                                                backward_view,
#                                                x11, y11, goal_line, x_min, x_max, y_min, y_max, person_radius,
#                                                step_length, push_dist, push_func)
#         else:
#             step_length = torch.sqrt((x12 - p0[0]) ** 2 + (y12 - p0[1]) ** 2)
#             new_state, failed = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
#                                                backward_view,
#                                                x12, y12, goal_line, x_min, x_max, y_min, y_max, person_radius,
#                                                step_length, push_dist, push_func)
#
#     else:  # Moving directly to goal
#         dx, dy = goal_x - p0[0], goal_y - p0[1]
#         goal_dist = torch.sqrt(dx ** 2 + dy ** 2)
#         dx, dy = step_length * dx / (goal_dist + EPS), step_length * dy / (goal_dist + EPS)
#         x1 = new_state[agent_identity, 0] + dx
#         y1 = new_state[agent_identity, 1] + dy
#         x1, y1 = stop_at_boundaries(p0[0], p0[1], x1, y1, x_min, x_max, y_min, goal_line, person_radius, EPS)
#         new_state, failed = update_state(new_state, x1, y1, agent_identity, variables_per_agent_per_timestep,
#                                          backward_view, person_radius, x_min, x_max, y_min, y_max, EPS)
#
#     return new_state, failed


def two_ways_evasion_heuristic(state: Tensor,
                               agent_identity: int,
                               evasion_func: Callable[[Tuple[Tensor]], Tuple[Tensor]],
                               which_side: str,
                               variables_per_agent_per_timestep: int,
                               backward_view: int,
                               goal_x: float,
                               goal_y: float,
                               goal_line: float,
                               x_min: float,
                               x_max: float,
                               y_min: float,
                               y_max: float,
                               person_radius: float,
                               step_length: float,
                               push: bool,
                               push_dist: float,
                               push_func: Callable[[object], object]) -> Tensor:
    new_state = state.clone()
    col_id, _ = collision_during_step(new_state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)
    p0 = state[agent_identity, :2].clone()
    del state

    if col_id >= 0:  # Evade
        pi = new_state[col_id, :2]
        x11, y11, x12, y12 = evasion_func(pi, p0, person_radius, step_length, EPS)
        x11, y11 = stop_at_boundaries(p0[0], p0[1], x11, y11, x_min, x_max, y_min, y_max, person_radius, EPS)
        x12, y12 = stop_at_boundaries(p0[0], p0[1], x12, y12, x_min, x_max, y_min, y_max, person_radius, EPS)
        if which_side is 'left':
            step_length = torch.sqrt((x11 - p0[0]) ** 2 + (y11 - p0[1]) ** 2)
            new_state, failed = step_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                               backward_view, x11, y11, goal_line, x_min, x_max, y_min,
                                               y_max, person_radius, step_length, push_dist, push_func)
        else:
            step_length = torch.sqrt((x12 - p0[0]) ** 2 + (y12 - p0[1]) ** 2)
            new_state, failed = step_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                               backward_view, x12, y12, goal_line, x_min, x_max, y_min,
                                               y_max, person_radius, step_length, push_dist, push_func)
        if failed is True:
            print('TRIED TO EVACUATE TO THE {} side but failed'.format(which_side))

    else:
        failed = True
        print('NO COLLIDER SO CAN NOT EVADE SIDE/TANGENT')

    return new_state, failed


def sideways_evasion_heuristic(state: Tensor,
                               agent_identity: int,
                               evasion_func: Callable[[Tuple[Tensor]], Tuple[Tensor]],
                               which_side: str,
                               variables_per_agent_per_timestep: int,
                               backward_view: int,
                               goal_x: float,
                               goal_y: float,
                               goal_line: float,
                               x_min: float,
                               x_max: float,
                               y_min: float,
                               y_max: float,
                               person_radius: float,
                               step_length: float,
                               push: bool,
                               push_dist: float,
                               push_func: Callable[[object], object]) -> Tensor:
    new_state = state.clone()
    col_id, _ = collision_during_step(new_state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)
    p0 = state[agent_identity, :2].clone()
    del state

    pi = new_state[col_id, :2]
    x11, y11, x12, y12 = evasion_func(pi, p0, person_radius, step_length, EPS)
    x11, y11 = stop_at_boundaries(p0[0], p0[1], x11, y11, x_min, x_max, y_min, y_max, person_radius, EPS)
    x12, y12 = stop_at_boundaries(p0[0], p0[1], x12, y12, x_min, x_max, y_min, y_max, person_radius, EPS)
    if which_side is 'left':
        step_length = torch.sqrt((x11 - p0[0]) ** 2 + (y11 - p0[1]) ** 2)
        new_state, failed = step_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                           backward_view, x11, y11, goal_line, x_min, x_max, y_min,
                                           y_max, person_radius, step_length, push_dist, push_func)
    else:
        step_length = torch.sqrt((x12 - p0[0]) ** 2 + (y12 - p0[1]) ** 2)
        new_state, failed = step_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                           backward_view, x12, y12, goal_line, x_min, x_max, y_min,
                                           y_max, person_radius, step_length, push_dist, push_func)
    if failed is True:
        print('TRIED TO EVACUATE TO THE {} side but failed'.format(which_side))

    return new_state, failed


def step_or_wait_heuristic(state: Tensor,
                           agent_identity: int,
                           variables_per_agent_per_timestep: int,
                           backward_view: int,
                           goal_x: float,
                           goal_y: float,
                           goal_line: float,
                           x_min: float,
                           # x_max: float,
                           y_min: float,
                           y_max: float,
                           person_radius: float,
                           step_length: float) -> Tensor:
    new_state = state.clone()
    del state
    col_id, _ = collision_during_step(new_state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)
    p0 = new_state[agent_identity, :2].clone()

    if col_id >= 0:  # wait
        new_state, failed = wait_heuristic(new_state, agent_identity, variables_per_agent_per_timestep, backward_view,
                                           person_radius, x_min, x_max, y_min, y_max, EPS)

    else:  # Moving directly to goal
        dx, dy = goal_x - p0[0], goal_y - p0[1]
        goal_dist = torch.sqrt(dx ** 2 + dy ** 2)
        dx, dy = step_length * dx / (goal_dist + EPS), step_length * dy / (goal_dist + EPS)
        x1 = new_state[agent_identity, 0] + dx
        y1 = new_state[agent_identity, 1] + dy
        x1, y1 = stop_at_boundaries(p0[0], p0[1], x1, y1, x_min, x_max, y_min, goal_line, person_radius, EPS)
        new_state, failed = update_state(new_state, x1, y1, agent_identity, variables_per_agent_per_timestep,
                                         backward_view, person_radius, x_min, x_max, y_min, y_max, EPS)

    return new_state, failed


def step_or_push_heuristic(state: Tensor,
                           agent_identity: int,
                           variables_per_agent_per_timestep: int,
                           backward_view: int,
                           goal_x: float,
                           goal_y: float,
                           goal_line: float,
                           x_min: float,
                           x_max: float,
                           y_min: float,
                           y_max: float,
                           person_radius: float,
                           step_length: float,
                           push_dist: float,
                           push_func: Callable[[object], object]) -> Tensor:
    col_id, _ = collision_during_step(state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)
    p0 = state[agent_identity, :2].clone()
    dx, dy = goal_x - p0[0], goal_y - p0[1]
    goal_dist = torch.sqrt(dx ** 2 + dy ** 2)
    dx, dy = step_length * dx / (goal_dist + EPS), step_length * dy / (goal_dist + EPS)
    x1 = state[agent_identity, 0] + dx
    y1 = state[agent_identity, 1] + dy
    x1, y1 = stop_at_boundaries(p0[0], p0[1], x1, y1, x_min, x_max, y_min, goal_line, person_radius, EPS)

    if col_id >= 0:  # Pushing in direction of goal
        new_state, failed = push_heuristic(state, agent_identity, variables_per_agent_per_timestep, backward_view,
                                           x1, y1, goal_line, x_min, x_max, y_min, y_max, person_radius,
                                           push_dist, push_dist, push_func)

    else:  # Moving directly to goal
        new_state, failed = update_state(state, x1, y1, agent_identity, variables_per_agent_per_timestep,
                                         backward_view, person_radius, x_min, x_max, y_min, y_max, EPS)

    del state

    return new_state, failed


def step_heuristic(state: Tensor,
                   agent_identity: int,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   goal_x: float,
                   goal_y: float,
                   goal_line: float,
                   x_min: float,
                   x_max: float,
                   y_min: float,
                   y_max: float,
                   person_radius: float,
                   step_length: float,
                   push_dist: float,
                   push_func: Callable[[object], object]) -> Tensor:
    col_id, _ = collision_during_step(state, agent_identity, person_radius, step_length, goal_x, goal_y, EPS)

    if col_id >= 0:  # Collision
        new_state = state.clone()
        failed = True
        print('TRIED TO STEP BUT COLLIDED')

    else:  # Moving directly to goal
        p0 = state[agent_identity, :2].clone()
        dx, dy = goal_x - p0[0], goal_y - p0[1]
        goal_dist = torch.sqrt(dx ** 2 + dy ** 2)
        dx, dy = step_length * dx / (goal_dist + EPS), step_length * dy / (goal_dist + EPS)
        x1 = state[agent_identity, 0] + dx
        y1 = state[agent_identity, 1] + dy
        x1, y1 = stop_at_boundaries(p0[0], p0[1], x1, y1, x_min, x_max, y_min, goal_line, person_radius, EPS)
        new_state, failed = update_state(state, x1, y1, agent_identity, variables_per_agent_per_timestep,
                                         backward_view, person_radius, x_min, x_max, y_min, y_max, EPS)
        if failed is True:
            print('STEP FAILED')

    del state

    return new_state, failed


def wait_heuristic(state: Tensor,
                   agent_identity: int,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   person_radius: float,
                   x_min: float,
                   x_max: float,
                   y_min: float,
                   y_max: float,
                   eps: float) -> Tensor:
    new_state, failed = update_state(state, state[agent_identity, 0], state[agent_identity, 1], agent_identity,
                                     variables_per_agent_per_timestep, backward_view, person_radius,
                                     x_min, x_max, y_min, y_max, EPS)
    return new_state, failed


def move_reduced(state: Tensor,
                 agent_identity: int,
                 action: Tensor,
                 person_radius: float,
                 soft_person_radius: float,
                 time_per_step: float,
                 variables_per_agent_per_timestep: int,
                 backward_view: int,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 goal_line: float,
                 step_length_runner: float,
                 runner_identities: List[int],
                 push_dist: float,
                 maximum_speed_forward: float,
                 maximum_speed_backward: float,
                 start_line: float,
                 **kwargs) -> Tensor:
    '''
    '''

    new_state = state.clone()
    del state
    right_axis = torch.tensor([1., 0.], device=new_state.device)
    if agent_identity in runner_identities:
        ref_point = torch.tensor([new_state[agent_identity, 0], goal_line], device=new_state.device)
    else:
        ref_point = right_axis
    shift_ref_point = ref_point - new_state[agent_identity, :2]
    ref_angle = angle_2D_full(shift_ref_point, right_axis)

    dv = action[0, 0]
    da = action[0, 1]

    velocity_old = torch.norm(new_state[agent_identity, :2] - new_state[agent_identity, 4:6])
    # angle_v_reference_line = angle_2D_full(velocity_old, shift_ref_point)
    # v_old = torch.norm(velocity_old) * torch.cos(angle_v_reference_line)
    # v = v_old + dv
    v = velocity_old + dv
    v = torch.clamp(v, maximum_speed_backward, maximum_speed_forward)

    a_old = angle_2D_full(new_state[agent_identity, 2:4], shift_ref_point)
    angle = a_old + da

    dx = time_per_step * v * torch.cos(ref_angle + angle)
    dy = time_per_step * v * torch.sin(ref_angle + angle)

    x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
    goal_x, goal_y = x0 + dx, y0 + dy

    new_state = update_state_without_collision_check(new_state, goal_x, goal_y, agent_identity,
                                                     variables_per_agent_per_timestep, backward_view, person_radius,
                                                     x_min, x_max, y_min, y_max, EPS)
    new_state[agent_identity, 2] = torch.cos(angle)
    new_state[agent_identity, 3] = torch.sin(angle)

    return new_state, False


def move(state: Tensor,
         agent_identity: int,
         action: Tensor,
         person_radius: float,
         soft_person_radius: float,
         time_per_step: float,
         variables_per_agent_per_timestep: int,
         backward_view: int,
         x_min: float,
         x_max: float,
         y_min: float,
         y_max: float,
         goal_line: float,
         step_length_runner: float,
         runner_identities: List[int],
         push_dist: float,
         maximum_speed_forward: float,
         maximum_speed_backward: float,
         start_line: float,
         **kwargs) -> Tensor:
    '''
    '''

    new_state = state.clone()
    try:
        del state

        v = action[0, 0]
        angle = action[0, 1]

        dx = time_per_step * v * torch.cos(angle)
        dy = time_per_step * v * torch.sin(angle)

        x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
        goal_x, goal_y = x0 + dx, y0 + dy

        goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, start_line, y_max, person_radius, EPS)

        step_length = torch.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)

        push_heur = do_not_push
        new_state, _ = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                      backward_view, goal_x, goal_y, goal_line, x_min, x_max, start_line,
                                      y_max, person_radius, step_length, push_dist, push_heur)

        new_state = update_state_without_collision_check(new_state, goal_x, goal_y, agent_identity,
                                                         variables_per_agent_per_timestep, backward_view, person_radius,
                                                         x_min, x_max, y_min, y_max, EPS)
        new_state[agent_identity, 2] = torch.cos(angle)
        new_state[agent_identity, 3] = torch.sin(angle)

    except Exception as e:
        print('ERROR MESAGE: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False


def move_pushable_for_heuristics(state: Tensor,
                                 agent_identity: int,
                                 action: Tensor,
                                 person_radius: float,
                                 soft_person_radius: float,
                                 time_per_step: float,
                                 variables_per_agent_per_timestep: int,
                                 backward_view: int,
                                 x_min: float,
                                 x_max: float,
                                 y_min: float,
                                 y_max: float,
                                 goal_line: float,
                                 step_length_runner: float,
                                 runner_identity: int,
                                 push_dist: float,
                                 **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''

    new_state = state.clone()
    direction_of_movement_x = torch.cos(action[0, 1])
    direction_of_movement_y = torch.sin(action[0, 1])
    dx = action[0, 0] * time_per_step * direction_of_movement_x
    dy = action[0, 0] * time_per_step * direction_of_movement_y

    x0, y0 = state[agent_identity, 0], state[agent_identity, 1]
    goal_x, goal_y = x0 + dx, y0 + dy
    goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, y_min, y_max, person_radius, EPS)
    dx, dy = goal_x - x0, goal_y - y0
    step_length = torch.sqrt(dx ** 2 + dy ** 2)

    new_state, failed = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                       backward_view, goal_x, goal_y, goal_line, x_min, x_max, y_min,
                                       y_max, person_radius, step_length, push_dist, collision_push)
    return new_state, failed


def heuristic_move(state: Tensor,
                   agent_identity: int,
                   action: Tensor,
                   person_radius: float,
                   soft_person_radius: float,
                   time_per_step: float,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   x_min: float,
                   x_max: float,
                   y_min: float,
                   y_max: float,
                   runner_goal_x: Tensor,
                   runner_goal_y: Tensor,
                   runner_identities: List[int],
                   step_length_runner: float,
                   step_length_waiting: float,
                   push_dist: float,
                   initial_state: Tensor,
                   maximum_speed: float,
                   **kwargs) -> Tensor:
    step_length_runner = time_per_step * step_length_runner
    try:
        if agent_identity in runner_identities:
            if torch.eq(action, torch.zeros_like(action)):
                new_state, failed = wait_heuristic(
                    state, agent_identity,
                    variables_per_agent_per_timestep, backward_view,
                    person_radius, x_min, x_max, y_min, y_max, EPS
                )
            elif torch.eq(action, 1 * torch.ones_like(action)):  # Step
                new_state, failed = step_heuristic(
                    state, agent_identity, variables_per_agent_per_timestep,
                    backward_view, state[agent_identity, 0], runner_goal_y,
                    runner_goal_y, x_min, x_max, y_min, y_max, person_radius,
                    step_length_runner, 0, do_not_push
                )
            elif torch.eq(action, 2 * torch.ones_like(action)):  # Try to evade tangentially to the left
                new_state, failed = two_ways_evasion_heuristic(
                    state, agent_identity, get_tangential_points, 'left',
                    variables_per_agent_per_timestep,
                    backward_view, state[agent_identity, 0],
                    runner_goal_y, runner_goal_y, x_min, x_max, y_min, y_max,
                    step_length=step_length_runner, person_radius=person_radius,
                    push=True, push_dist=push_dist, push_func=do_not_push
                )
            elif torch.eq(action, 3 * torch.ones_like(action)):  # Try to evade tangentially to the right
                new_state, failed = two_ways_evasion_heuristic(
                    state, agent_identity, get_tangential_points, 'right',
                    variables_per_agent_per_timestep,
                    backward_view, state[agent_identity, 0],
                    runner_goal_y, runner_goal_y, x_min, x_max, y_min, y_max,
                    step_length=step_length_runner, person_radius=person_radius,
                    push=True, push_dist=push_dist, push_func=do_not_push
                )
            elif torch.eq(action, 4 * torch.ones_like(action)):  # Try to evade to the left
                new_state, failed = sideways_evasion_heuristic(
                    state, agent_identity, get_lateral_points, 'left',
                    variables_per_agent_per_timestep,
                    backward_view, state[agent_identity, 0],
                    runner_goal_y, runner_goal_y, x_min, x_max, y_min, y_max,
                    step_length=step_length_runner / 2., person_radius=person_radius,
                    push=True, push_dist=push_dist, push_func=do_not_push
                )
            elif torch.eq(action, 5 * torch.ones_like(action)):  # Try to evade to the right
                new_state, failed = sideways_evasion_heuristic(
                    state, agent_identity, get_lateral_points, 'right',
                    variables_per_agent_per_timestep,
                    backward_view, state[agent_identity, 0],
                    runner_goal_y, runner_goal_y, x_min, x_max, y_min, y_max,
                    step_length=step_length_runner / 2., person_radius=person_radius,
                    push=True, push_dist=push_dist, push_func=do_not_push
                )
            # elif torch.eq(action, 6 * torch.ones_like(action)):  # Push
            #     # print('Push')
            #     new_state, failed = push_move_heuristic(state, agent_identity, variables_per_agent_per_timestep,
            #                                             backward_view,
            #                                             state[agent_identity, 0], runner_goal_y, runner_goal_y,
            #                                             x_min, x_max, y_min, y_max, person_radius, step_length_runner,
            #                                             push_dist, side_push)

            if bool(torch.eq(action, torch.zeros_like(action))) is False and \
                    check_movement(new_state, agent_identity, variables_per_agent_per_timestep, backward_view) is False:
                new_state, failed = state.clone(), True


        else:
            new_state, failed = move_pushable_movement_adaption(
                state, agent_identity, action, person_radius,
                soft_person_radius, time_per_step,
                variables_per_agent_per_timestep, backward_view,
                x_min, x_max, y_min, y_max,
                runner_goal_y, step_length_runner,
                runner_identities, 0, maximum_speed
            )
    except Exception as e:
        print('Error message: {}'.format(e))
        return state, True

    return new_state, failed


def move_pushable(state: Tensor,
                  agent_identity: int,
                  action: Tensor,
                  person_radius: float,
                  soft_person_radius: float,
                  time_per_step: float,
                  variables_per_agent_per_timestep: int,
                  backward_view: int,
                  x_min: float,
                  x_max: float,
                  y_min: float,
                  y_max: float,
                  goal_line: float,
                  step_length_runner: float,
                  runner_identities: List[int],
                  push_dist: float,
                  **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''
    new_state = state.clone()
    try:
        del state
        v = action[0, 0]
        angle = action[0, 1]
        dx = time_per_step * v * torch.cos(angle)
        dy = time_per_step * v * torch.sin(angle)

        x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
        goal_x, goal_y = x0 + dx, y0 + dy
        goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, y_min, y_max, person_radius, EPS)
        step_length = torch.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)

        if bool(v > EPS and step_length < EPS) is True:
            failed = True
            return new_state, failed

        push_heur = do_not_push
        # if agent_identity in runner_identities:
        #     push_heur = collision_push
        new_state, _ = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                      backward_view, goal_x, goal_y, goal_line, x_min, x_max, y_min,
                                      y_max, person_radius, step_length, push_dist, push_heur)

        step_length = torch.sqrt((new_state[agent_identity, 0] - x0) ** 2 + (new_state[agent_identity, 1] - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            failed = True
            return new_state, failed

    except Exception as e:
        print('ERROR MESAGE: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False


def move_pushable(state: Tensor,
                  agent_identity: int,
                  action: Tensor,
                  person_radius: float,
                  soft_person_radius: float,
                  time_per_step: float,
                  variables_per_agent_per_timestep: int,
                  backward_view: int,
                  x_min: float,
                  x_max: float,
                  y_min: float,
                  y_max: float,
                  goal_line: float,
                  step_length_runner: float,
                  runner_identities: List[int],
                  push_dist: float,
                  start_line: float,
                  **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''
    new_state = state.clone()
    try:
        del state
        v = action[0, 0]
        angle = action[0, 1]
        dx = time_per_step * v * torch.cos(angle)
        dy = time_per_step * v * torch.sin(angle)

        x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
        goal_x, goal_y = x0 + dx, y0 + dy
        goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, start_line, y_max, person_radius, EPS)
        step_length = torch.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)

        if bool(v > EPS and step_length < EPS) is True:
            failed = True
            return new_state, failed

        push_heur = do_not_push
        # if agent_identity in runner_identities:
        #     push_heur = collision_push
        new_state, _ = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                      backward_view, goal_x, goal_y, goal_line, x_min, x_max, start_line,
                                      y_max, person_radius, step_length, push_dist, push_heur)

        step_length = torch.sqrt((new_state[agent_identity, 0] - x0) ** 2 + (new_state[agent_identity, 1] - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            failed = True
            return new_state, failed

    except Exception as e:
        print('ERROR MESAGE: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False


def move_pushable_movement_adaption(state: Tensor,
                                    agent_identity: int,
                                    action: Tensor,
                                    person_radius: float,
                                    soft_person_radius: float,
                                    time_per_step: float,
                                    variables_per_agent_per_timestep: int,
                                    backward_view: int,
                                    x_min: float,
                                    x_max: float,
                                    y_min: float,
                                    y_max: float,
                                    goal_line: float,
                                    step_length_runner: float,
                                    runner_identities: List[int],
                                    push_dist: float,
                                    maximum_speed_forward: float,
                                    maximum_speed_backward: float,
                                    start_line: float,
                                    **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''
    new_state = state.clone()
    try:
        del state

        push_heur = do_not_push
        # if agent_identity in runner_identities:
        #     # push_heur = collision_push
        #     goal = new_state[agent_identity, 0:2].clone()
        #     goal[1] = goal_line
        # else:
        #     goal = new_state[agent_identity, 4:6].clone()

        # Update velocity module
        v = action[0, 0]

        # Update angle
        old_angle = angle_2D_full(new_state[agent_identity, 2:4], torch.tensor([1., 0.], device=new_state.device))
        angle = old_angle + action[0, 1]

        dx = time_per_step * v * torch.cos(angle)
        dy = time_per_step * v * torch.sin(angle)

        x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
        goal_x, goal_y = x0 + dx, y0 + dy

        goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, start_line, y_max, person_radius, EPS)
        step_length = torch.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            new_state[agent_identity, 2] = torch.cos(angle)
            new_state[agent_identity, 3] = torch.sin(angle)
            failed = True
            return new_state, failed

        new_state, _ = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                      backward_view, goal_x, goal_y, goal_line, x_min, x_max, start_line,
                                      y_max, person_radius, step_length, push_dist, push_heur)

        step_length = torch.sqrt((new_state[agent_identity, 0] - x0) ** 2 + (new_state[agent_identity, 1] - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            new_state[agent_identity, 2] = torch.cos(angle)
            new_state[agent_identity, 3] = torch.sin(angle)
            failed = True
            return new_state, failed

        new_state[agent_identity, 2] = torch.cos(angle)
        new_state[agent_identity, 3] = torch.sin(angle)


    except Exception as e:
        print('ERROR MESAGE: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False


def move_pushable_movement_adaption2(state: Tensor,
                                     agent_identity: int,
                                     action: Tensor,
                                     person_radius: float,
                                     soft_person_radius: float,
                                     time_per_step: float,
                                     variables_per_agent_per_timestep: int,
                                     backward_view: int,
                                     x_min: float,
                                     x_max: float,
                                     y_min: float,
                                     y_max: float,
                                     goal_line: float,
                                     step_length_runner: float,
                                     runner_identities: List[int],
                                     push_dist: float,
                                     maximum_speed_forward: float,
                                     maximum_speed_backward: float,
                                     start_line: float,
                                     **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''
    new_state = state.clone()
    try:
        del state

        push_heur = do_not_push

        old_v = torch.norm(new_state[agent_identity, 4:6] - new_state[agent_identity, :2])
        old_angle = angle_2D_full(new_state[agent_identity, 2:4], torch.tensor([1., 0.], device=new_state.device))

        new_v = torch.norm(action[0])
        new_angle = angle_2D_full(action[0], torch.tensor([1., 0.], device=new_state.device))

        momentum = 0.9
        v = momentum * old_v + (1 - momentum) * new_v
        angle = momentum * old_angle + (1 - momentum) * new_angle

        # Force speed to [0, maximum_speed]
        v = torch.clamp(v, maximum_speed_backward, maximum_speed_forward)

        dx = time_per_step * v * torch.cos(angle)
        dy = time_per_step * v * torch.sin(angle)

        x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
        goal_x, goal_y = x0 + dx, y0 + dy

        goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, start_line, y_max, person_radius, EPS)
        step_length = torch.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            new_state[agent_identity, 2] = torch.cos(angle)
            new_state[agent_identity, 3] = torch.sin(angle)
            failed = True
            return new_state, failed

        new_state, _ = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                      backward_view, goal_x, goal_y, goal_line, x_min, x_max, start_line,
                                      y_max, person_radius, step_length, push_dist, push_heur)

        step_length = torch.sqrt((new_state[agent_identity, 0] - x0) ** 2 + (new_state[agent_identity, 1] - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            new_state[agent_identity, 2] = torch.cos(angle)
            new_state[agent_identity, 3] = torch.sin(angle)
            failed = True
            return new_state, failed

        new_state[agent_identity, 2] = torch.cos(angle)
        new_state[agent_identity, 3] = torch.sin(angle)


    except Exception as e:
        print('ERROR MESAGE: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False


def move_pushable_continous(state: Tensor,
                            agent_identity: int,
                            action: Tensor,
                            person_radius: float,
                            soft_person_radius: float,
                            time_per_step: float,
                            variables_per_agent_per_timestep: int,
                            backward_view: int,
                            x_min: float,
                            x_max: float,
                            y_min: float,
                            y_max: float,
                            goal_line: float,
                            step_length_runner: float,
                            runner_identity: int,
                            push_dist: float,
                            **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''

    new_state = state.clone()
    dx, dy = action[0], action[1]

    x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
    goal_x, goal_y = x0 + dx, y0 + dy
    goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, y_min, y_max, person_radius, EPS)
    dx, dy = goal_x - x0, goal_y - y0
    step_length = torch.sqrt(dx ** 2 + dy ** 2)

    new_state, failed = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                       backward_view, goal_x, goal_y, goal_line, x_min, x_max, y_min,
                                       y_max, person_radius, step_length, push_dist, side_push)

    del state, step_length, x0, y0, goal_x, goal_y, dx, dy

    return new_state, failed


def move_pushable_mixed(state: Tensor,
                        agent_identity: int,
                        action: Tensor,
                        person_radius: float,
                        soft_person_radius: float,
                        time_per_step: float,
                        variables_per_agent_per_timestep: int,
                        backward_view: int,
                        x_min: float,
                        x_max: float,
                        y_min: float,
                        y_max: float,
                        goal_line: float,
                        step_length_runner: float,
                        runner_identity: int,
                        push_dist: float,
                        step_lengths_runner: List[float],
                        step_lengths_waiting: List[float],
                        **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''

    new_state = state.clone()
    del state
    angle = action[0, 0].clone()
    if agent_identity == runner_identity:
        step_length = step_lengths_runner[int(action[0, 1])]
    else:
        step_length = step_lengths_waiting[int(action[0, 1])]

    direction_of_movement_x = torch.cos(angle)
    direction_of_movement_y = torch.sin(angle)
    dx = step_length * time_per_step * direction_of_movement_x
    dy = step_length * time_per_step * direction_of_movement_y

    x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
    goal_x, goal_y = x0 + dx, y0 + dy
    goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, y_min, y_max, person_radius, EPS)
    dx, dy = goal_x - x0, goal_y - y0
    step_length = torch.sqrt(dx ** 2 + dy ** 2)

    new_state, failed = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                       backward_view, goal_x, goal_y, goal_line, x_min, x_max, y_min,
                                       y_max, person_radius, step_length, push_dist, side_push)
    return new_state, failed


def move_martinez_gil(state: Tensor,
                      agent_identity: int,
                      action: Tensor,
                      person_radius: float,
                      soft_person_radius: float,
                      time_per_step: float,
                      variables_per_agent_per_timestep: int,
                      backward_view: int,
                      x_min: float,
                      x_max: float,
                      y_min: float,
                      y_max: float,
                      goal_line: float,
                      step_length_runner: float,
                      runner_identities: List[int],
                      push_dist: float,
                      maximum_speed: float,
                      start_line: float,
                      **kwargs) -> Tensor:
    '''
        Moves the pedestrian and allows to push others away.
    '''
    new_state = state.clone()
    try:
        del state

        push_heur = do_not_push
        if agent_identity in runner_identities:
            # push_heur = collision_push
            goal = new_state[agent_identity, 0:2].clone()
            goal[1] = goal_line
        else:
            goal = new_state[agent_identity, 4:6].clone()

        # Update velocity module
        shift = new_state[agent_identity, :2] - new_state[agent_identity, 4:6]
        old_v = torch.norm(shift) / time_per_step
        v = old_v + torch.clamp(action[0, 0], -1.75, 1.75)

        # Force speed to [0, maximum_speed]
        v = torch.clamp(v, 0., maximum_speed)

        # Angle of reference line to positive x_axis
        dir_ref_line = goal - new_state[agent_identity, :2]
        dir_x_positive = torch.tensor([1., 0.], device=new_state.device)
        angle_ref_line = angle_2D_full(dir_ref_line, dir_x_positive)

        # Update angle relative to reference line
        old_angle = angle_2D_full(new_state[agent_identity, 2:4], dir_ref_line)
        angle = old_angle + torch.clamp(action[0, 1], -math_pi / 4.,
                                        math_pi / 4.)  # angle relative to the reference line

        # Convert angle back to positive x axis
        angle = angle + angle_ref_line

        # # Force angle to [0, 2pi]
        # if bool(angle > 2 * math_pi):
        #     angle = angle - 2 * math_pi
        # elif bool(angle < 0):
        #     angle = 2 * math_pi + angle

        dx = time_per_step * v * torch.cos(angle)
        dy = time_per_step * v * torch.sin(angle)

        x0, y0 = new_state[agent_identity, 0], new_state[agent_identity, 1]
        goal_x, goal_y = x0 + dx, y0 + dy

        goal_x, goal_y = stop_at_boundaries(x0, y0, goal_x, goal_y, x_min, x_max, start_line, y_max, person_radius, EPS)
        step_length = torch.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            failed = True
            return new_state, failed

        new_state, _ = push_heuristic(new_state, agent_identity, variables_per_agent_per_timestep,
                                      backward_view, goal_x, goal_y, goal_line, x_min, x_max, start_line,
                                      y_max, person_radius, step_length, push_dist, push_heur)

        step_length = torch.sqrt((new_state[agent_identity, 0] - x0) ** 2 + (new_state[agent_identity, 1] - y0) ** 2)
        if bool(v > EPS and step_length < EPS) is True:
            failed = True
            return new_state, failed

        new_state[agent_identity, 2] = torch.cos(angle)
        new_state[agent_identity, 3] = torch.sin(angle)


    except Exception as e:
        print('ERROR MESAGE: ', e)
        # failed = True

    # for i in range(new_state.shape[0]):
    #     if check_out_of_bounds(new_state, i, x_min, x_max, y_min, y_max):
    #         print('BOUNDARY COLLISION')

    return new_state, False
