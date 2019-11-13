from typing import List, Tuple

from math import pi as math_pi

import numpy

import torch
from torch import Tensor

from pg_ped.utils import distance_matrix, break_if_nan


def cut_goal_line(y, goal_line):
    if y >= goal_line:
        y = goal_line
    return y


def check_agent_passed_goal_line(state: Tensor,
                                 agent_identity: int,
                                 goal_line: float,
                                 person_radius: float) -> bool:
    position = state[agent_identity, :2]
    y_position = position[1]
    person_radius_t = torch.tensor(person_radius, device=state.device)
    goal_line_t = torch.tensor(goal_line, device=state.device)
    agent_passed_goal_line = bool((y_position + person_radius_t >= goal_line_t))
    del person_radius_t, goal_line_t
    return agent_passed_goal_line


def check_personal_space(state: Tensor, agent_identity: int, person_radius: float, eps: float,
                         variables_per_agent_per_timestep: int):
    previous_position = state[agent_identity,
                        variables_per_agent_per_timestep: variables_per_agent_per_timestep + 2]
    indices = [i for i in range(state.shape[0]) if i != agent_identity]
    positions = state[indices, variables_per_agent_per_timestep: variables_per_agent_per_timestep + 2]
    positions = torch.cat([previous_position.unsqueeze(0), positions])

    dists = distances_for_collision_checking(positions, agent_identity, person_radius)
    colliders = dists < (2 * person_radius)
    is_collision = numpy.any(colliders)
    if bool(is_collision) is True:
        return True, numpy.where(colliders == True)[0][0]
    else:
        return False, -1


def check_agent_collision(state: Tensor, agent_identity: int, person_radius: float, eps: float):
    positions = state[:, :2]
    dists = distances_for_collision_checking(positions, agent_identity, person_radius)
    colliders = dists < (2 * person_radius)
    is_collision = numpy.any(colliders)
    if bool(is_collision) is True:
        return True, numpy.where(colliders == True)[0][0]
    else:
        return False, -1


def check_agent_collision_all(state: Tensor, person_radius: float, eps: float):
    collider = -1
    for i in range(state.shape[0]):
        collision, collision_with = check_agent_collision(state, i, person_radius, eps)
        if collision is True:
            collider = i
            check_agent_collision(state, i, person_radius, eps)
            break
    return collision, [collider, collision_with]


def push_at_boundaries(x0: Tensor, y0: Tensor, x1: Tensor, y1: Tensor, x_min, x_max: float, y_min: float, y_max: float,
                       person_radius: float, eps: float):
    '''
    Try stop at boundary. If that leads to no almost no movement, shift along boundary.

    :param x0:
    :param y0:
    :param x1:
    :param y1:
    :param x_min:
    :param y_min:
    :param x_max:
    :param y_max:
    :param person_radius:
    :return:
    '''
    device = x0.device
    if bool(x1 < x_min + person_radius):
        s = (x_min - x0 + person_radius) / (x1 - x0 + eps)
        x1 = x_min + person_radius
        if s >= 1e-5:
            y1 = y0 + s * (y1 - y0)
    elif bool(x1 > x_max - person_radius):
        s = (x_max - x0 - person_radius) / (x1 - x0 + eps)
        x1 = x_max - person_radius
        if s >= 1e-5:
            y1 = y0 + s * (y1 - y0)
    elif bool(y1 < y_min + person_radius + eps):
        s = (y_min - y0 + person_radius) / (y1 - y0 + eps)
        y1 = y_min + person_radius
        if s >= 1e-5:
            x1 = x0 + s * (x1 - x0)
    elif bool(y1 > y_max - person_radius):
        s = (y_max - y0 - person_radius) / (y1 - y0 + eps)
        y1 = y_max - person_radius
        if s >= 1e-5:
            x1 = x0 + s * (x1 - x0)
    return x1, y1


def stop_at_boundaries(x0: Tensor, y0: Tensor, x1: Tensor, y1: Tensor, x_min, x_max: float, y_min: float, y_max: float,
                       person_radius: float, eps: float):
    '''
        Stops movement of a circle from (x0, y0) to (x1, y1) at the rectangle boundaries.
        It is assumed that the distance from start to end is not greater than the radius
        of the circle. Otherwise it would have to be determined which boundary is
        intersected first.
    '''
    device = x0.device
    if bool(x1 < x_min + person_radius):
        if bool(x1 == x0):
            s = 0
        else:
            s = (x_min - x0 + person_radius) / (x1 - x0)
        x1 = torch.tensor(x_min + person_radius, device=x1.device)
        y1 = y0 + s * (y1 - y0)

    if bool(x1 > x_max - person_radius):
        if bool(x1 == x0):
            s = 0
        else:
            s = (x_max - x0 - person_radius) / (x1 - x0)
        x1 = torch.tensor(x_max - person_radius, device=x1.device)
        y1 = y0 + s * (y1 - y0)

    if bool(y1 < y_min + person_radius):
        if bool(y0 == y1):
            s = 0
        else:
            s = (y_min - y0 + person_radius) / (y1 - y0)
        y1 = torch.tensor(y_min + person_radius, device=x1.device)
        x1 = x0 + s * (x1 - x0)

    if bool(y1 > y_max - person_radius):
        if bool(y0 == y1):
            s = 0
        else:
            s = (y_max - y0 - person_radius) / (y1 - y0)
        y1 = torch.tensor(y_max - person_radius, device=x1.device)
        x1 = x0 + s * (x1 - x0)
    return x1, y1


def cut_boundaries(x, y, x_min, y_min, x_max, y_max, person_radius):
    device = x.device
    if x < x_min + person_radius:
        x = torch.tensor(x_min + person_radius, device=device)
    if x > x_max - person_radius:
        x = torch.tensor(x_max - person_radius, device=device)
    if y < y_min + person_radius:
        y = torch.tensor(y_min + person_radius, device=device)
    if y > y_max - person_radius:
        y = torch.tensor(y_max - person_radius, device=device)
    return x, y


def cut_agent_intersection(state: Tensor, agent_identity: int, dx: Tensor, dy: Tensor, person_radius: float):
    '''
        This function does not work as intended in the moment!
        A workaround was used in the move function: If a movement leads to a collision, nothing is done instead.
        Make sure that two pedestrian circles do not overlap. This function uses equation of the form y=mx + t,
        so there will always be some directional error, because moving straightforward would require m->inf.

    :param state:
    :param agent_identity:
    :param dx:
    :param dy:
    :return:
    '''

    p0 = state[agent_identity, :2]
    x0, y0 = p0[0], p0[1]
    x1, y1 = x0 + dx, y0 + dy
    m = dy / (dx + 1e-12)  # prevent division by zero
    t = y0 - m * x0

    # Compute endpoints of half-plane
    x_shift = person_radius / torch.sqrt(m ** 2 + 1)
    x_ext11, x_ext12 = x0 - x_shift, x0 + x_shift
    y_ext11, y_ext12 = m * x_ext11 + t, m * x_ext12 + t
    if (x1 - x_ext11) ** 2 + (y1 - y_ext11) ** 2 > (x1 - x0) ** 2 + (
            y1 - y0) ** 2:  # Choose a solution of quadr. equation
        x_ext1, y_ext1 = x_ext11, y_ext11
        x_ext2 = x1 + x_shift
        y_ext2 = m * x_ext2 + t
    else:
        x_ext1, y_ext1 = x_ext12, y_ext12
        x_ext2 = x1 - x_shift
        y_ext2 = m * x_ext2 + t

    mn0 = mn1 = -m
    tn0, tn1 = y_ext1 + (y_ext1 - t), y_ext2 + (y_ext2 - t)

    s_min = 1
    for i in range(state.shape[0]):
        if i != agent_identity:

            pi = state[i, :2]
            xi, yi = pi[0], pi[1]

            if (yi > xi * mn0 + tn0) and (
                    yi < xi * mn1 + tn1):  # Pedestrian obstacle is in plane bounded by orthogonals on endpoints of linesection

                ml = -m
                tl = yi - ml * xi

                # Point of intersection
                x_intersect = (tl - t) / 2.
                y_intersect = ml * x_intersect + tl
                dl = (x_intersect - xi) ** 2 + (y_intersect - yi) ** 2
                dl = torch.sqrt(dl)

                if dl < person_radius:  # Pedestrian obstacle is so close to the trajectory, that the pedestrians will collide
                    vsqr = dx ** 2 + dy ** 2
                    a = 2 * (x0 - xi + y0 - yi)
                    discr = a ** 2 - 4 * vsqr * ((x0 - xi) ** 2 + (y0 - yi) ** 2 - 4 * person_radius ** 2)
                    sqrt_discr = torch.sqrt(discr)
                    s1 = (a + sqrt_discr) / (2 * vsqr)
                    s2 = (a - sqrt_discr) / (2 * vsqr)

                    if s1 >= 0:
                        if s1 < s_min:
                            s_min = s1
                    if s1 < 0:
                        if s2 < s_min:
                            s_min = s2

    x0, y0 = x0 + s_min * (x1 - x0), y0 + s_min * (y1 - y0)
    return x0, y0


def check_boundary_collision(state: Tensor,
                             agent_identity: int,
                             x_min: float, x_max: float,
                             y_min: float, y_max: float,
                             person_radius: float,
                             **kwargs):
    box_t = torch.tensor([x_min, x_max, y_min, y_max], device=state.device)
    x = state[agent_identity, 0]
    y = state[agent_identity, 1]
    if bool(x - person_radius < box_t[0] or \
            x + person_radius > box_t[1] or \
            y - person_radius < box_t[2] or \
            y + person_radius > box_t[3]):
        return True
    else:
        return False


def check_out_of_bounds(state: Tensor,
                        agent_identity: int,
                        x_min: float, x_max: float,
                        y_min: float, y_max: float,
                        **kwargs):
    box_t = torch.tensor([x_min, x_max, y_min, y_max], device=state.device)
    x = state[agent_identity, 0]
    y = state[agent_identity, 1]
    if bool(x <= box_t[0] or \
            x >= box_t[1] or \
            y <= box_t[2] or \
            y >= box_t[3]):
        return True
    else:
        return False


def distances_for_collision_checking(positions: Tensor,
                                     agent_identity: int,
                                     person_radius: float):
    dists = distance_matrix(positions)
    dists = dists[agent_identity, :]
    dists[agent_identity] = dists.max() + 2 * person_radius
    return dists


def nearest_neighbor_distance(positions: Tensor,
                              agent_identity: int,
                              indices: List[int],
                              person_radius: float) -> Tuple[int, Tensor]:
    dists = distances_for_collision_checking(positions, agent_identity, person_radius)
    sorted_indices = numpy.argsort(dists)
    nearest = indices[sorted_indices[0]]
    distance_to_nearest = dists[sorted_indices[0]]
    return nearest, distance_to_nearest


def identify_agent_collision(state: Tensor,
                             agent_identity: int,
                             person_radius: float) -> float:
    '''
    :param state:
    :param agent_identity:
    :param person_radius:
    :return: Index into state with which the agent will collide first.
    '''

    positions = state[:, :2]
    indices = [i for i in range(positions.shape[0])]
    nearest, distance_to_nearest = nearest_neighbor_distance(positions, agent_identity, indices, person_radius)
    if distance_to_nearest < 2 * (person_radius - 1e-5):
        return nearest
    else:
        return -1  # is empty


def stretch_points(state: Tensor, factor: Tensor):
    '''
    This function moves all points just a little bit, such that the distances between them
    increases so slightly, that collision due to rounding errors are avoided.

    :param state:
    :param agent_identity:
    :return:
    '''

    center = state[:, :2].mean(dim=0)
    shifts_to_center = state[:, :2] - center
    stretched = center + (1 + factor) * shifts_to_center
    new_state = state.clone()
    new_state[:, :2] = stretched
    return new_state


def identify_circles_in_stripe(state: Tensor,
                               agent_identity: int,
                               indices: List[int],
                               p0: Tensor,
                               p1: Tensor,
                               extension_1: float,
                               extension_2: float,
                               eps: float) -> List[int]:
    '''
        Searches for indices into state of circles which are in an area between two parallel lines, which are
        orthogonal to the shift vector p1 - p0 and pass through the extended endpoints of the line sector
        from p0 to p1.
    '''
    delta_p = p1 - p0
    v = torch.norm(delta_p, p=2)

    potential_colliders = []
    for i in indices:

        if i != agent_identity:

            pi = state[i, :2]

            def extension_line_foot(x, y):
                return y + x * (delta_p[0] / (delta_p[1] + eps)) - \
                       p0[1] + (extension_1 / (v + eps)) * delta_p[1] + (delta_p[0] / (delta_p[1] + eps)) * \
                       ((extension_1 / (v + eps)) * delta_p[0] - p0[0])

            def extension_line_peak(x, y):
                return y + x * (delta_p[0] / (delta_p[1] + eps)) - \
                       p1[1] - (extension_2 / (v + eps)) * delta_p[1] - (delta_p[0] / (delta_p[1] + eps)) * \
                       ((extension_2 / (v + eps)) * delta_p[0] + p1[0])

            foot, peak = extension_line_foot(pi[0], pi[1]), extension_line_peak(pi[0], pi[1])

            if bool(foot < -1e-5 and peak > 1e-5) is True:
                potential_colliders += [i]

            if bool(foot > 1e-5 and peak < -1e-5) is True:
                potential_colliders += [i]

    return potential_colliders


def intersection_line_with_line(p0: Tensor, p1: Tensor, pi: Tensor, eps: float) -> Tensor:
    '''
    Intersection of the line spanned by p0 and p1 with an orthogonal through pi.

    :param p0:
    :param p1:
    :param pi:
    :return:
    '''

    dp = p1 - p0
    dx, dy = dp[0], dp[1]
    dq = torch.tensor([dy, -dx], device=p0.device)
    xi, yi = pi[0], pi[1]
    x0, y0 = p0[0], p0[1]

    s_inter = -(dy * (y0 - yi) + dx * (x0 - xi)) / (dx ** 2 + dy ** 2 + eps)
    p_inter = p0 + s_inter * dp

    # m = dp[1] / (dp[0] + eps)
    # t = p0[1] - m * p0[0]
    # x_inter = (m / (m ** 2 + 1)) * (pi[1] + (pi[0] / (m + eps)) - (p0[1] - m * p0[0]))
    # y_inter = -(1 / (m + eps)) * x_inter + pi[1] + pi[0] / (m + eps)
    return p_inter


def distance_to_line(p0: Tensor, p1: Tensor, pi: Tensor, eps: float) -> Tensor:
    '''
    Distance of pi to linesegment going from p0 to p1.

    :param p0:
    :param p1:
    :param pi:
    :return:
    '''

    p_inter = intersection_line_with_line(p0, p1, pi, eps)
    dist = torch.norm(p_inter - pi, p=2)
    return dist


def rotate_vector_by_angle(vector: Tensor, angle: float) -> Tuple[Tensor]:
    x0, y0 = vector[0], vector[1]
    x1 = torch.cos(angle) * x0 - torch.sin(angle) * y0
    y1 = torch.sin(angle) * x0 + torch.cos(angle) * y0
    return x1, y1


def angle_2D(v1: Tensor, v2: Tensor):
    v1_normed = v1 / torch.norm(v1)
    v2_normed = v2 / torch.norm(v2)
    return torch.acos(torch.dot(v1_normed, v2_normed))


def angle_2D_full(v1: Tensor, v2: Tensor):
    '''
        Compute the angle between v1 and v2 in mathematically positive direction of rotation
        angle_2D_full: R² x R² -> [0, 2pi]
        v2 is the reference
    '''
    x_plus_dir = torch.tensor([1., 0.], device=v1.device)
    y_plus_dir = torch.tensor([0., 1.], device=v1.device)

    x_half1 = torch.dot(v1, x_plus_dir)
    y_half1 = torch.dot(v1, y_plus_dir)

    x_half2 = torch.dot(v2, x_plus_dir)
    y_half2 = torch.dot(v2, y_plus_dir)

    # cases v1
    if bool(x_half1 >= 0) and bool(y_half1 >= 0):  # 1. Quadrant
        theta_x1 = torch.atan(v1[1] / (v1[0] + 1e-8))
    elif bool(x_half1 < 0) and bool(y_half1 >= 0):  # 2. Quadrant
        theta_x1 = math_pi - torch.atan(v1[1] / (-v1[0] + 1e-8))
    elif bool(x_half1 < 0) and bool(y_half1 <= 0):  # 3. Quadrant
        theta_x1 = math_pi + torch.atan(torch.abs(v1[1] / (v1[0] + 1e-8)))
    elif bool(x_half1 >= 0) and bool(y_half1 < 0):  # 4. Quadrant
        theta_x1 = 2 * math_pi - torch.atan(- v1[1] / (v1[0] + 1e-8))

    # cases v2
    if bool(x_half2 >= 0) and bool(y_half2 >= 0):  # 1. Quadrant
        theta_x2 = torch.atan(v2[1] / (v2[0] + 1e-8))
    elif bool(x_half2 < 0) and bool(y_half2 >= 0):  # 2. Quadrant
        theta_x2 = math_pi - torch.atan(v2[1] / (-v2[0] + 1e-8))
    elif bool(x_half2 < 0) and bool(y_half2 <= 0):  # 3. Quadrant
        theta_x2 = math_pi + torch.atan(torch.abs(v2[1] / (v2[0] + 1e-8)))
    elif bool(x_half2 >= 0) and bool(y_half2 < 0):  # 4. Quadrant
        theta_x2 = 2 * math_pi - torch.atan(- v2[1] / (v2[0] + 1e-8))

    # compute angle between v1 and v2
    theta = theta_x1 - theta_x2
    if bool(theta < 0):
        theta = 2 * math_pi + theta

    return theta


def get_lateral_points(pi: Tensor, p0: Tensor,
                       person_radius: float,
                       step_length: float, eps: float) -> Tuple[Tensor]:
    '''
    Computes the points the are shifted orthogonally with respect to the shift vector from p0 to pi.

    :param pi:
    :param p0:
    :param person_radius:
    :param step_length:
    :return:
    '''

    dp = pi - p0
    dp_orthogonal = torch.tensor([-dp[1], dp[0]], device=pi.device)
    v = torch.norm(dp_orthogonal)

    step_length = step_length / 3.  # decrease step size because it is slower to move laterally
    p11 = p0 + dp_orthogonal * step_length / (v + eps)
    p12 = p0 - dp_orthogonal * step_length / (v + eps)

    break_if_nan(torch.cat([p11, p12]))
    return p11[0], p11[1], p12[0], p12[1]


def get_tangential_points(pi: Tensor, p0: Tensor,
                          person_radius: float,
                          step_length: float, eps: float) -> Tuple[Tensor]:
    '''
    Compute the points at which a circle moving on a line through p0 and a circle around pi touch.

    :param pi:
    :param p0:
    :param person_radius:
    :param step_length:
    :return:
    '''

    step_length = step_length / 2.  # decrease stepsize because it is slower to move tangentially

    theta = torch.asin((2 * person_radius) / torch.norm(pi - p0, p=2))
    if bool(torch.isnan(theta)):
        theta = torch.tensor(math_pi / 2., device=pi.device)
    dx1, dy1 = rotate_vector_by_angle(pi - p0, theta)
    v1 = torch.sqrt(dx1 ** 2 + dy1 ** 2)
    x11, y11 = p0[0] + dx1 * (step_length / (v1 + 1e-8)), p0[1] + dy1 * (step_length / (v1 + 1e-8))

    dx2, dy2 = rotate_vector_by_angle(pi - p0, -theta)
    v2 = torch.sqrt(dx2 ** 2 + dy2 ** 2)
    x12, y12 = p0[0] + dx2 * (step_length / (v2 + 1e-8)), p0[1] + dy2 * (step_length / (v2 + 1e-8))

    all = torch.tensor([x11, y11, x12, y12])
    break_if_nan(all)
    return x11, y11, x12, y12


def get_tangential_points_vadere(p0: Tensor,
                                 person_radius: float,
                                 eps: float):
    '''
    Translated from java-code:
    https://gitlab.lrz.de/vadere/vadere/blob/master/VadereSimulator/src/org/vadere/simulator/models/bhm/UtilsBHM.java
    
    :return: 
    '''

    x0, y0 = p0[0], p0[1]
    radius_sqr = person_radius ** 2

    # cases for numerical stability
    if bool(y0 < eps and y0 > -eps):

        x1 = radius_sqr / x0
        x2 = x1
        y1 = torch.sqrt(radius_sqr - x1 ** 2)
        y2 = -y1

    elif bool(x0 < eps and x0 > -eps):

        y1 = radius_sqr / y0
        y2 = y1
        x1 = torch.sqrt(radius_sqr - y1 ** 2)
        x2 = -x1

    elif bool(x0 < y0):

        y0_sqr = y0 ** 2

        a = 1 + (x0 ** 2) / y0_sqr
        b = -2 * radius_sqr * x0 / y0_sqr
        c = (radius_sqr ** 2 / y0_sqr) - radius_sqr

        discriminant = torch.sqrt(b ** 2 - 4 * a * c)

        x1 = (-b + discriminant) / (2 * a)
        x2 = (-b - discriminant) / (2 * a)
        y1 = (radius_sqr - x1 * x0) / y0
        y2 = (radius_sqr - x2 * x0) / y0

    else:

        x0_sqr = y0 ** 2

        a = (1 + y0 ** 2 / x0_sqr)
        b = -(2 * radius_sqr * (y0 / x0_sqr))
        c = (radius_sqr ** 2 / x0_sqr) - radius_sqr

        discriminant = torch.sqrt(b ** 2 - 4 * a * c)

        y1 = (-b + discriminant) / (2 * a)
        y2 = (-b - discriminant) / (2 * a)
        x1 = (radius_sqr - y1 * y0) / x0
        x2 = (radius_sqr - y2 * y0) / x0

    return x1, y1, x2, y2


def collision_during_step(state: Tensor,
                          agent_identity: int,
                          person_radius: float,
                          step_length: float,
                          goal_x: float,
                          goal_y: float,
                          eps: float) -> int:
    '''
        Detect if during a step from current position to goal_x, goal_y, an agent (circle) collides with another
        agent (circle) and compute the distance between initial center of moving circle and the stopping point
    '''

    number_agents = int(state.shape[0])
    p0 = state[agent_identity, :2]
    p_goal = torch.tensor([goal_x, goal_y], device=state.device)
    delta_p = p_goal - p0
    v = torch.norm(delta_p, p=2)
    step_goal = p0 + delta_p * (step_length / (v + eps))
    orthogonal_delta_p = torch.tensor([-delta_p[1], delta_p[0]], device=state.device)
    q0 = p0 + (person_radius / (v + eps)) * orthogonal_delta_p
    q1 = p0 - (person_radius / (v + eps)) * orthogonal_delta_p

    potential_colliders = identify_circles_in_stripe(state, agent_identity, [i for i in range(number_agents)],
                                                     p0, step_goal, 0., 0., eps)
    colliders = identify_circles_in_stripe(state, agent_identity, potential_colliders, q0, q1,
                                           person_radius, person_radius, eps)
    for i in range(number_agents):
        if i != agent_identity:
            dist_p0 = torch.norm(state[i, :2] - p0)
            dist_step_goal = torch.norm(state[i, :2] - step_goal)
            if bool(dist_p0 < 2 * (person_radius - 1e-5)) or bool(dist_step_goal < 2 * (person_radius - 1e-5)):
                colliders += [i]

            del dist_step_goal, dist_p0

    if len(colliders) == 0:
        nearest_collider = -1
        dist = numpy.nan
    else:
        positions = state[[agent_identity] + colliders, :2]
        nearest_collider, distance_to_nearest_collider = nearest_neighbor_distance(positions, 0,
                                                                                   [agent_identity] + colliders,
                                                                                   person_radius)
        position_collider = positions[([agent_identity] + colliders).index(nearest_collider)]
        nearest_collider_distance_to_step = distance_to_line(p0, p_goal, position_collider, eps)
        p_inter = intersection_line_with_line(p0, p_goal,
                                              position_collider, eps)
        step_back_from_intercept = torch.sqrt(
            torch.abs(4 * person_radius ** 2 - nearest_collider_distance_to_step ** 2))
        p1 = p_inter - delta_p * (step_back_from_intercept / (v + eps))
        dist = torch.norm(p1 - p0, p=2)
        break_if_nan(dist)

        del positions, position_collider, p_inter, p1

    del p0, p_goal, delta_p, v, step_goal, orthogonal_delta_p, q0, q1

    return nearest_collider, dist


def collision_during_step2(state: Tensor,
                           agent_identity: int,
                           person_radius: float,
                           eps: float) -> int:
    '''
        Detect if during a step from the last position to current position, an agent (circle) collides with another
        agent (circle) and compute the distance between initial center of moving circle and the stopping point
    '''

    number_agents = int(state.shape[0])
    p0 = state[agent_identity, 4:6]
    p_goal = state[agent_identity, :2]
    delta_p = p_goal - p0
    v = torch.norm(delta_p)
    step_goal = p0 + delta_p
    orthogonal_delta_p = torch.tensor([-delta_p[1], delta_p[0]], device=state.device)
    q0 = p0 + (person_radius / (v + eps)) * orthogonal_delta_p
    q1 = p0 - (person_radius / (v + eps)) * orthogonal_delta_p

    potential_colliders = identify_circles_in_stripe(state, agent_identity, [i for i in range(number_agents)],
                                                     p0, step_goal, 0., 0., eps)
    colliders = identify_circles_in_stripe(state, agent_identity, potential_colliders, q0, q1,
                                           person_radius, person_radius, eps)
    for i in range(number_agents):
        if i != agent_identity:
            dist_p0 = torch.norm(state[i, :2] - p0)
            dist_step_goal = torch.norm(state[i, :2] - step_goal)
            if bool(dist_p0 < 2 * (person_radius - 1e-5)) or bool(dist_step_goal < 2 * (person_radius - 1e-5)):
                colliders += [i]

            del dist_step_goal, dist_p0

    if len(colliders) == 0:
        nearest_collider = -1
        dist = numpy.nan
    else:
        positions = state[[agent_identity] + colliders, :2]
        nearest_collider, distance_to_nearest_collider = nearest_neighbor_distance(positions, 0,
                                                                                   [agent_identity] + colliders,
                                                                                   person_radius)
        position_collider = positions[([agent_identity] + colliders).index(nearest_collider)]
        nearest_collider_distance_to_step = distance_to_line(p0, p_goal, position_collider, eps)
        p_inter = intersection_line_with_line(p0, p_goal,
                                              position_collider, eps)
        step_back_from_intercept = torch.sqrt(
            torch.abs(4 * person_radius ** 2 - nearest_collider_distance_to_step ** 2))
        p1 = p_inter - delta_p * (step_back_from_intercept / (v + eps))
        dist = torch.norm(p1 - p0, p=2)
        break_if_nan(dist)

        del positions, position_collider, p_inter, p1

    del p0, p_goal, delta_p, v, step_goal, orthogonal_delta_p, q0, q1

    return nearest_collider, dist


def do_not_push(xi0: Tensor,
                yi0: Tensor,
                x1: Tensor,
                y1: Tensor,
                dx: Tensor,
                dy: Tensor,
                t: float,
                v: float,
                push_dist: float,
                eps: float,
                variables_per_agent_per_timestep: int,
                backward_view: int,
                **kwargs) -> Tuple[Tensor]:
    return xi0, yi0


def collision_push(xi0: Tensor,
                   yi0: Tensor,
                   x1: Tensor,
                   y1: Tensor,
                   dx: Tensor,
                   dy: Tensor,
                   t: float,
                   v: float,
                   push_dist: float,
                   eps: float,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   **kwargs) -> Tuple[Tensor]:
    shift_x = xi0 - x1
    shift_y = yi0 - y1
    shift = torch.sqrt(shift_x ** 2 + shift_y ** 2)
    xi1, yi1 = xi0 + shift_x * push_dist / (shift + eps), yi0 + shift_y * push_dist / (shift + eps)
    return xi1, yi1


def move_dir_push(xi0: Tensor,
                  yi0: Tensor,
                  x1: Tensor,
                  y1: Tensor,
                  dx: Tensor,
                  dy: Tensor,
                  t: float,
                  v: float,
                  push_dist: float,
                  eps: float,
                  variables_per_agent_per_timestep: int,
                  backward_view: int,
                  **kwargs) -> Tuple[Tensor]:
    xi1, yi1 = xi0 + dx * push_dist / (v + eps), yi0 + dy * push_dist / (v + eps)
    return xi1, yi1


def side_push(xi0: Tensor,
              yi0: Tensor,
              x1: Tensor,
              y1: Tensor,
              dx: float,
              dy: float,
              t: float,
              v: float,
              push_dist: float,
              eps: float,
              variables_per_agent_per_timestep: int,
              backward_view: int,
              **kwargs) -> Tuple[Tensor]:
    side_of_runner_goal_line = yi0 - ((dy / (dx + eps)) * xi0 + t)
    if side_of_runner_goal_line < torch.zeros_like(side_of_runner_goal_line):
        xi1 = xi0 + push_dist * dy / (v + eps)
        yi1 = yi0 - push_dist * dx / (v + eps)
    else:
        xi1 = xi0 - push_dist * dy / (v + eps)
        yi1 = yi0 + push_dist * dx / (v + eps)

    return xi1, yi1


def seperate_overlapping(state: Tensor, person_radius: float,
                         x_min: float, x_max: float, y_min: float, y_max: float, eps: float,
                         variables_per_agent_per_timestep: int, backward_view: int, **kwargs) -> Tensor:
    new_state = state.clone()
    del state
    positions = new_state[:, :2]
    collision, collision_pair = check_agent_collision_all(new_state, person_radius, eps)
    d0 = torch.tensor(3 * person_radius, device=new_state.device)
    d1 = torch.tensor(4 * person_radius, device=new_state.device)
    i = 0
    while collision is True and bool(torch.abs(d1 - d0) > 1e-5) is True:
        i += 1
        coll_pos = positions[collision_pair].clone()
        stretch_factor = 1e-3 + ((2 * person_radius) / torch.norm(coll_pos[1] - coll_pos[0])) - 1
        coll_pos_stretched = stretch_points(coll_pos, stretch_factor)
        x10, y10 = coll_pos[0, 0], coll_pos[0, 1]
        x20, y20 = coll_pos[1, 0], coll_pos[1, 1]
        x11, y11 = coll_pos_stretched[0, 0], coll_pos_stretched[0, 1]
        x21, y21 = coll_pos_stretched[1, 0], coll_pos_stretched[1, 1]
        x11, y11 = stop_at_boundaries(x10, y10, x11, y11, x_min, x_max, y_min, y_max, person_radius, eps)
        x21, y21 = stop_at_boundaries(x20, y20, x21, y21, x_min, x_max, y_min, y_max, person_radius, eps)
        d0 = torch.norm(new_state[collision_pair[0], :2] - new_state[collision_pair[1], :2])
        new_state[collision_pair[0], 0], new_state[collision_pair[0], 1] = x11, y11
        new_state[collision_pair[1], 0], new_state[collision_pair[1], 1] = x21, y21
        d1 = torch.norm(new_state[collision_pair[0], :2] - new_state[collision_pair[1], :2])
        collision, collision_pair = check_agent_collision_all(new_state, person_radius, eps)
        positions = new_state[:, :2]
        # if i > 10:
        #     print('Long loop')

    failed = False
    if check_agent_collision_all(new_state, person_radius, eps)[0] is True:
        failed = True

    return new_state, failed


def update_state(state: Tensor,
                 x1: Tensor, y1: Tensor,
                 agent_identity: int,
                 variables_per_agent_per_timestep: int,
                 backward_view: int,
                 person_radius: float,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 eps: float):
    new_state = state.clone()
    new_state[agent_identity, variables_per_agent_per_timestep:variables_per_agent_per_timestep * backward_view] = \
        state[agent_identity, :variables_per_agent_per_timestep * (backward_view - 1)]
    new_state[agent_identity, 0] = x1
    new_state[agent_identity, 1] = y1
    new_state[agent_identity, 2] = x1 - state[agent_identity, 0]
    new_state[agent_identity, 3] = y1 - state[agent_identity, 1]

    # Free memory
    del x1, y1

    # Make sure that no overlapping of agents can occur. If that is not possible, stay at the old state.
    new_state, failed = seperate_overlapping(new_state, person_radius,
                                             x_min, x_max, y_min, y_max, eps,
                                             variables_per_agent_per_timestep, backward_view)
    if failed is True:
        new_state = state.clone()
        print('SEPERATION FAILED')
    del state

    # Detect nans
    failed = break_if_nan(new_state) or failed
    if break_if_nan(new_state):
        print('FAILED BECAUSE OF NANS')

    return new_state, failed


def update_state_without_collision_check(state: Tensor,
                                         x1: Tensor, y1: Tensor,
                                         agent_identity: int,
                                         variables_per_agent_per_timestep: int,
                                         backward_view: int,
                                         person_radius: float,
                                         x_min: float,
                                         x_max: float,
                                         y_min: float,
                                         y_max: float,
                                         eps: float,
                                         **kwargs):
    new_state = state.clone()
    new_state[agent_identity, variables_per_agent_per_timestep:variables_per_agent_per_timestep * backward_view] = \
        state[agent_identity, :variables_per_agent_per_timestep * (backward_view - 1)]
    new_state[agent_identity, 0] = x1
    new_state[agent_identity, 1] = y1
    new_state[agent_identity, 2] = x1 - state[agent_identity, 0]
    new_state[agent_identity, 3] = y1 - state[agent_identity, 1]

    # Free memory
    del x1, y1

    return new_state


def check_step_size(state: Tensor,
                    agent_identity: int,
                    variables_per_agent_per_timestep: int,
                    backward_view: int,
                    **kwargs):
    p0 = state[agent_identity, variables_per_agent_per_timestep:variables_per_agent_per_timestep + 2]
    p1 = state[agent_identity, :2]
    shift = p1 - p0
    dist = torch.norm(shift)
    return dist


def check_movement(state: Tensor,
                   agent_identity: int,
                   variables_per_agent_per_timestep: int,
                   backward_view: int,
                   tol: float = 1e-7,
                   **kwargs):
    dist = check_step_size(state, agent_identity, variables_per_agent_per_timestep, backward_view)
    movement = bool(dist > tol)
    return movement
