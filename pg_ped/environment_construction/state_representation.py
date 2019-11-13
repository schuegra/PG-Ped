import torch
from torch import Tensor

from pg_ped.marl.utils import *


def render_localization_map(state: Tensor, agent_identity: int,
                            x_min: float, x_max: float, y_min: float, y_max: float,
                            variables_per_agent_per_timestep: int, backward_view: int, rows: int, cols: int,
                            device: str, person_radius: float, **kwargs):
    localization_map = torch.zeros(1, backward_view, rows, cols, device=device)
    for t in range(backward_view):
        if bool(state[agent_identity, t * variables_per_agent_per_timestep + 1] >= y_min) is True:
            row, col = cart_to_img(state[agent_identity, t * variables_per_agent_per_timestep],
                                   state[agent_identity, t * variables_per_agent_per_timestep + 1],
                                   x_min, x_max, y_min, y_max, rows, cols, person_radius)
            try:
                localization_map[0, t, row, col] += 1
            except IndexError:
                print('IndexError while rendering heat map.')
                cart_to_img(state[agent_identity, t * variables_per_agent_per_timestep],
                            state[agent_identity, t * variables_per_agent_per_timestep + 1],
                            x_min, x_max, y_min, y_max, rows, cols, person_radius)
            except Exception:
                print('Other Error while rendering heat map.')
                cart_to_img(state[agent_identity, t * variables_per_agent_per_timestep],
                            state[agent_identity, t * variables_per_agent_per_timestep + 1],
                            x_min, x_max, y_min, y_max, rows, cols, person_radius)

    return localization_map


def render_heat_map(state: Tensor,
                    x_min: float,
                    x_max: float,
                    y_min: float,
                    y_max: float,
                    variables_per_agent_per_timestep: int,
                    backward_view: int,
                    rows: int,
                    cols: int,
                    device: str,
                    person_radius: float,
                    **kwargs):
    number_agents = state.shape[0]
    heat_map = torch.zeros(1, backward_view, rows, cols, device=device)
    for i in range(number_agents):
        heat_map += render_localization_map(state, i, x_min, x_max, y_min, y_max,
                                            variables_per_agent_per_timestep, backward_view, rows, cols,
                                            device, person_radius, **kwargs)

    heat_map[0, :, 0, :cols - 1] += 1.
    heat_map[0, :, rows - 1, 1:] += 1.
    heat_map[0, :, 1:, 0] += 1.
    heat_map[0, :, rows - 1:, cols - 1] += 1.

    return heat_map


def render_gaussian_density(state: Tensor,
                            x_min: float,
                            x_max: float,
                            y_min: float,
                            y_max: float,
                            variables_per_agent_per_timestep: int,
                            backward_view: int,
                            rows: int,
                            cols: int,
                            device: str,
                            person_radius: float,
                            standard_deviation: float,
                            influence_radius: float,
                            **kwargs):
    heat_map = render_heat_map(state, x_min, x_max, y_min, y_max, variables_per_agent_per_timestep, backward_view,
                               rows, cols, device, person_radius)
    gaussian_filter = get_gaussian_kernel(standard_deviation, influence_radius, backward_view, device)
    density_map = gaussian_filter(heat_map)
    density_map = normalize_tensor(density_map)
    return density_map


def render_gd_and_loc_map(state: Tensor,
                          agent_identity: int,
                          x_min: float,
                          x_max: float,
                          y_min: float,
                          y_max: float,
                          variables_per_agent_per_timestep: int,
                          backward_view: int,
                          rows: int,
                          cols: int,
                          device: str,
                          person_radius: float,
                          standard_deviation: float,
                          influence_radius: float,
                          **kwargs):
    localization_map = render_localization_map(state, agent_identity, x_min, x_max, y_min, y_max,
                                               variables_per_agent_per_timestep, backward_view,
                                               rows, cols, device, person_radius)
    gaussian_density = render_gaussian_density(state, x_min, x_max, y_min, y_max,
                                               variables_per_agent_per_timestep, backward_view,
                                               rows, cols, device, person_radius, standard_deviation,
                                               influence_radius)
    gaussian_density_and_localization_map = torch.cat([gaussian_density, localization_map], dim=1)

    return gaussian_density_and_localization_map


def render_local_gd(state: Tensor,
                    agent_identity: int,
                    x_min: float,
                    x_max: float,
                    y_min: float,
                    y_max: float,
                    variables_per_agent_per_timestep: int,
                    backward_view: int,
                    rows: int,
                    cols: int,
                    window_rows: int,
                    window_cols: int,
                    device: str,
                    person_radius: float,
                    standard_deviation: float,
                    influence_radius: float,
                    **kwargs):
    gaussian_density_and_localization_map = render_gd_and_loc_map(state, agent_identity, x_min, x_max, y_min, y_max,
                                                                  variables_per_agent_per_timestep, backward_view,
                                                                  rows, cols, device, person_radius, standard_deviation,
                                                                  influence_radius)
    u, v = cart_to_img(state[agent_identity, 0], state[agent_identity, 1],
                       x_min, x_max, y_min, y_max, rows, cols, person_radius)
    window_radius_height = int((window_rows - 1) / 2.)
    window_radius_width = int((window_cols - 1) / 2.)
    if u - window_radius_height < 0:
        u = window_radius_height + 1
    if u + window_radius_height >= rows:
        u = rows - window_radius_height - 1
    if v - window_radius_width < 0:
        v = window_radius_width + 1
    if v + window_radius_width >= cols:
        v = cols - window_radius_width - 1

    local_gd = gaussian_density_and_localization_map[:, :backward_view,
               u - window_radius_height: u + window_radius_height + 1,
               v - window_radius_width:v + window_radius_width + 1]
    local_loc = gaussian_density_and_localization_map[:, backward_view:,
                u - window_radius_height: u + window_radius_height + 1,
                v - window_radius_width:v + window_radius_width + 1]
    local_gd_loc = torch.cat([local_gd, local_loc], dim=1)

    return local_gd_loc


def render_scenario(state: Tensor,
                    agent_identity: int,
                    x_min: float,
                    x_max: float,
                    y_min: float,
                    y_max: float,
                    goal_line: float,
                    variables_per_agent_per_timestep: int,
                    backward_view: int,
                    rows: int,
                    cols: int,
                    window_rows: int,
                    window_cols: int,
                    device: str,
                    person_radius: float,
                    standard_deviation: float,
                    influence_radius: float,
                    **kwargs):
    # Render scenario
    number_persons = state.shape[0]
    color_current = 0.1
    color_others = 0.2
    color_wall = 'r'
    color_goal = 'b'

    row_radius, col_radius = int((window_rows - 1) / 2), int((window_cols - 1) / 2)
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    extended_scenario_array = numpy.zeros([ext_scen_size_row, ext_scen_size_col, 3])
    my_dpi = 96.
    agent_windows = numpy.zeros([3 * backward_view, window_rows, window_cols])
    for i in range(backward_view):
        fig, ax = plt.subplots(figsize=[cols / my_dpi, rows / my_dpi], dpi=my_dpi)
        init_axes(ax, x_min, x_max, y_min, y_max, goal_line, color_wall, color_goal)
        vis_state(
            state[:,
            i * variables_per_agent_per_timestep: (i + 1) * variables_per_agent_per_timestep] \
                .detach().cpu().numpy(),
            [color_current if i == agent_identity else color_others for i in range(number_persons)],
            ax, fig, person_radius, x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols,
            device, standard_deviation, with_arrows=False, **kwargs
        )
        scenario_array = fig_to_data(fig)
        plt.close()
        scenario_array = (scenario_array / scenario_array.max())
        extended_scenario_array[row_radius:ext_scen_size_row - row_radius, col_radius:ext_scen_size_col - col_radius,
        :] = scenario_array

        # Cut window around agent
        u, v = cart_to_img(state[agent_identity, i * variables_per_agent_per_timestep], state[agent_identity, 1],
                           x_min, x_max, y_min, y_max, rows, cols, person_radius)
        u, v = int(u.cpu().numpy()), int(v.cpu().numpy())
        agent_window = extended_scenario_array[u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1), :]
        try:
            agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window.transpose(2, 0, 1)
        except:
            print('FAIL')

    agent_windows = torch.tensor(agent_windows, device=device).float().unsqueeze(0)
    return agent_windows


def render_vision_field(state: Tensor,
                        agent_identity: int,
                        x_min: float,
                        x_max: float,
                        y_min: float,
                        y_max: float,
                        goal_line: float,
                        variables_per_agent_per_timestep: int,
                        backward_view: int,
                        rows: int,
                        cols: int,
                        window_rows: int,
                        window_cols: int,
                        device: str,
                        person_radius: float,
                        standard_deviation: float,
                        influence_radius: float,
                        ax: plt.Axes,
                        fig=plt.Figure,
                        colors=numpy.ndarray,
                        **kwargs):
    # Render scenario
    state_numpy = state.cpu().numpy()
    del state
    number_persons = state_numpy.shape[0]
    color_wall = 'r'
    color_goal = 'b'

    row_radius, col_radius = int((window_rows - 1.) / 2), int((window_cols - 1.) / 2)
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    # + 40 because the runner has negative y-coordinates in his initial positions
    extended_scenario_array = numpy.zeros([ext_scen_size_row + 40, ext_scen_size_col, 3])
    agent_windows = numpy.zeros([3 * backward_view, window_rows, window_cols])

    for i in range(backward_view):
        init_axes(ax, x_min, x_max, y_min, y_max, goal_line, color_wall, color_goal)
        vis_state(
            state_numpy[:, i * variables_per_agent_per_timestep: (i + 1) * variables_per_agent_per_timestep],
            colors, ax, fig, person_radius, x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols,
            device, standard_deviation, with_arrows=False, with_init_axes=False, **kwargs
        )
        scenario_array = fig_to_data(fig)
        scenario_array = (scenario_array * 1. / scenario_array.max())

        # Coordinates of agent
        uv = cart_to_img_numpy(
            state_numpy[agent_identity, i * variables_per_agent_per_timestep],
            state_numpy[agent_identity, i * variables_per_agent_per_timestep + 1],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )

        # Mask field of view
        fov_radius = row_radius if window_rows < window_cols else col_radius
        scenario_array *= numpy.repeat(create_circular_mask(rows, cols, uv, fov_radius), 3).reshape(rows, cols, 3)

        # Fill into extended scenario
        extended_scenario_array[row_radius:ext_scen_size_row - row_radius, col_radius:ext_scen_size_col - col_radius,
        :] = scenario_array

        # Cut window around agent
        agent_window = extended_scenario_array[uv[0]:(uv[0] + 2 * row_radius + 1), uv[1]:(uv[1] + 2 * col_radius + 1),
                       :]

        agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window.transpose(2, 0, 1)
    agent_windows = torch.tensor(agent_windows, device=device).float().unsqueeze(0)
    # vis_feature_maps(agent_windows)
    return agent_windows


def render_full_scenario_torch(state: Tensor,
                               agent_identity: int,
                               x_min: float,
                               x_max: float,
                               y_min: float,
                               y_max: float,
                               goal_line: float,
                               variables_per_agent_per_timestep: int,
                               backward_view: int,
                               rows: int,
                               cols: int,
                               window_rows: int,
                               window_cols: int,
                               device: str,
                               person_radius: float,
                               standard_deviation: float,
                               influence_radius: float,
                               runner_identities: List[int],
                               start_line: float,
                               **kwargs):
    entrance_line = y_min - start_line
    y_min = 0
    y_max = y_max - start_line
    goal_line = goal_line - start_line

    # Render scenario
    new_state = state.clone()
    for a in range(new_state.shape[0]):
        new_state[a, 1] = new_state[a, 1] - start_line
        new_state[a, 1 + 4] = new_state[a, 1 + 4] - start_line
        new_state[a, 1 + 2 * 4] = new_state[a, 1 + 2 * 4] - start_line
    del state
    number_persons = new_state.shape[0]
    color_wall = torch.tensor([0, 0, 1], device=device).float()
    color_goal = torch.tensor([0, 1, 0], device=device).float()
    color_others = torch.tensor([1, 0, 0], device=device).float()
    color_current = torch.tensor([1, 1, 1], device=device).float()
    color_reference_line = torch.tensor([0.5, 0.5, 0.5], device=device).float()

    row_radius, col_radius = torch.tensor((window_rows - 1.) / 2, device=device).int(), \
                             torch.tensor((window_cols - 1.) / 2, device=device).int()
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    scenarios = torch.zeros([3 * backward_view, max(rows, cols), max(rows, cols)], device=device)
    for i in range(backward_view):
        scenario_array = scenario_torch(
            new_state[:, i * variables_per_agent_per_timestep: i * variables_per_agent_per_timestep + 2],
            agent_identity, color_current, color_others, color_wall, color_goal, person_radius,
            x_min, x_max, y_min, y_max, goal_line, entrance_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, device
        )

        # Add reference lines
        u, v = cart_to_img_torch(
            new_state[agent_identity, i * variables_per_agent_per_timestep:
                                      i * variables_per_agent_per_timestep + 2],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        if agent_identity in runner_identities:
            scenario_array[:u, v] += color_reference_line
        else:
            goal = new_state[agent_identity, 2 * 4: 2 * 4 + 2]  # second last position
            u_goal, v_goal = cart_to_img_torch(goal, x_min, x_max, y_min, y_max, rows, cols, person_radius)
            v_start, v_end = v, v_goal
            u_start, u_end = u, u_goal
            start = torch.cat([u_start.unsqueeze(0), v_start.unsqueeze(0)]).float()
            end = torch.cat([u_end.unsqueeze(0), v_end.unsqueeze(0)]).float()
            dir = end - start
            length = torch.norm(dir).ceil()
            steps = torch.arange(length).int().to(device)
            ts = steps.float() / length
            line = torch.zeros(length.int(), 2, device=device)
            for t, step in zip(ts, steps):
                line[step, :] = start + t * dir
            line = line.round().long()
            line_rows = line[:, 0]
            line_columns = line[:, 1]
            scenario_array[line_rows, line_columns] += color_reference_line

        scenario_array = scenario_array.permute(2, 0, 1)

        pixel_diff = abs(rows - cols)
        if rows > cols:
            pad_top = pad_bottom = 0
            if pixel_diff % 2 == 0:
                pad_left = pad_right = int(pixel_diff / 2)
            else:
                pad_left = int(pixel_diff / 2.) + 1
                pad_right = int(pixel_diff / 2.)
        else:
            pad_left = pad_right = 0
            if pixel_diff % 2 == 0:
                pad_top = pad_bottom = int(pixel_diff / 2)
            else:
                pad_top = int(pixel_diff / 2.) + 1
                pad_bottom = int(pixel_diff / 2.)
        scenario_padded = pad_image(scenario_array, padding=(pad_left, pad_top, pad_right, pad_bottom))

        scenarios[3 * i: 3 * (i + 1), :, :] = scenario_padded

    scenarios = scenarios.unsqueeze(0)
    # vis_feature_maps(agent_windows)

    return scenarios


def render_rectangular_fov_torch(state: Tensor,
                                 agent_identity: int,
                                 x_min: float,
                                 x_max: float,
                                 y_min: float,
                                 y_max: float,
                                 goal_line: float,
                                 variables_per_agent_per_timestep: int,
                                 backward_view: int,
                                 rows: int,
                                 cols: int,
                                 window_rows: int,
                                 window_cols: int,
                                 device: str,
                                 person_radius: float,
                                 standard_deviation: float,
                                 influence_radius: float,
                                 runner_identities: List[int],
                                 start_line: float,
                                 **kwargs):
    entrance_line = y_min - start_line
    y_min = 0
    y_max = y_max - start_line
    goal_line = goal_line - start_line

    # Render scenario
    new_state = state.clone()
    for a in range(new_state.shape[0]):
        new_state[a, 1] = new_state[a, 1] - start_line
        new_state[a, 1 + 4] = new_state[a, 1 + 4] - start_line
        new_state[a, 1 + 2 * 4] = new_state[a, 1 + 2 * 4] - start_line
    del state
    number_persons = new_state.shape[0]
    color_wall = torch.tensor([0, 0, 1], device=device).float()
    color_goal = torch.tensor([0, 1, 0], device=device).float()
    color_others = torch.tensor([1, 0, 0], device=device).float()
    color_current = torch.tensor([1, 1, 1], device=device).float()
    color_reference_line = torch.tensor([0.5, 0.5, 0.5], device=device).float()

    row_radius, col_radius = torch.tensor((window_rows - 1.) / 2, device=device).int(), \
                             torch.tensor((window_cols - 1.) / 2, device=device).int()
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    # + 40 / + 120 because the runner has negative y-coordinates in his initial positions
    extended_scenario_array = 0.2 * torch.ones([3, ext_scen_size_row + 120, ext_scen_size_col])
    agent_windows = torch.zeros([3 * backward_view, window_rows, window_cols], device=device)

    for i in range(backward_view):
        scenario_array = scenario_torch(
            new_state[:, i * variables_per_agent_per_timestep: i * variables_per_agent_per_timestep + 2],
            agent_identity, color_current, color_others, color_wall, color_goal, person_radius,
            x_min, x_max, y_min, y_max, goal_line, entrance_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, device
        )

        # Coordinates and movement of agent
        u, v = cart_to_img_torch(
            new_state[agent_identity, i * variables_per_agent_per_timestep:
                                      i * variables_per_agent_per_timestep + 2],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        looking_in_cartesian_direction = new_state[agent_identity, i * variables_per_agent_per_timestep:
                                                                   i * variables_per_agent_per_timestep + 2] + \
                                         new_state[agent_identity, i * variables_per_agent_per_timestep + 2:
                                                                   i * variables_per_agent_per_timestep + 4]
        u_lookdirimg, v_lookdirimg = cart_to_img_torch(
            looking_in_cartesian_direction, x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        du, dv = u_lookdirimg - u, v_lookdirimg - v

        # To ensure that the agent circle is included
        if x_max - x_min > y_max - y_min:
            person_radius_px = torch.tensor(rows * person_radius * 1. / (y_max - y_min), device=device).int()
        else:
            person_radius_px = torch.tensor(cols * person_radius * 1. / (x_max - x_min), device=device).int()
        shift_length = torch.sqrt(du.float() ** 2 + dv.float() ** 2)
        if shift_length == 0.:
            shift_length = shift_length + person_radius_px
        normalized_du, normalized_dv = du.float() / shift_length, dv.float() / shift_length
        if normalized_du > 0:
            normalized_du = normalized_du.ceil()
        else:
            normalized_du = normalized_du.floor()
        if normalized_dv > 0:
            normalized_dv = normalized_dv.ceil()
        else:
            normalized_dv = normalized_dv.floor()

        # Add reference lines
        if agent_identity in runner_identities:
            scenario_array[:u, v] += color_reference_line
        else:
            goal = new_state[agent_identity, 2 * 4: 2 * 4 + 2]  # second last position
            u_goal, v_goal = cart_to_img_torch(goal, x_min, x_max, y_min, y_max, rows, cols, person_radius)
            v_start, v_end = v, v_goal
            u_start, u_end = u, u_goal
            start = torch.cat([u_start.unsqueeze(0), v_start.unsqueeze(0)]).float()
            end = torch.cat([u_end.unsqueeze(0), v_end.unsqueeze(0)]).float()
            dir = end - start
            length = torch.norm(dir).ceil()
            steps = torch.arange(length).int().to(device)
            ts = steps.float() / length
            line = torch.zeros(length.int(), 2, device=device)
            for t, step in zip(ts, steps):
                line[step, :] = start + t * dir
            line = line.round().long()
            line_rows = line[:, 0]
            line_columns = line[:, 1]
            scenario_array[line_rows, line_columns] += color_reference_line

        # plt.imshow(scenario_array.cpu().numpy());
        # plt.show()

        # Fill into extended scenario
        extended_scenario_array[:, row_radius:ext_scen_size_row - row_radius,
        col_radius:ext_scen_size_col - col_radius] = scenario_array.permute(2, 0, 1)

        # Rotate
        angle = angle_2D_full(  # angle of orientation in [0, 2pi]
            new_state[agent_identity, i * variables_per_agent_per_timestep + 2:
                                      i * variables_per_agent_per_timestep + 4],
            torch.tensor([0., -1.], device=device)  # in row direction (negative y-axis -> positive row-axis)
        )
        extended_scenario_array_rotated = rotate_image(extended_scenario_array, 2 * math_pi - angle,
                                                       center_u=u + row_radius,
                                                       center_v=v + col_radius)

        # plt.imshow(extended_scenario_array_rotated.cpu().numpy().transpose(1,2,0));
        # plt.show()

        # Cut window around agent
        # agent_window = extended_scenario_array_rotated[:, u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1)]
        agent_window = cut_window(extended_scenario_array_rotated, window_rows, window_cols)
        agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window

        # plt.imshow(agent_window.cpu().numpy().transpose(1,2,0));
        # plt.show()
        #
        # print('OK')

    agent_windows = agent_windows.unsqueeze(0)
    # vis_feature_maps(agent_windows)

    # Clean up
    del color_wall, color_goal, color_others, color_current, row_radius, col_radius, scenario_array, u, v  # , fov_radius
    del extended_scenario_array, agent_window

    return agent_windows


def render_fov_no_rotation(state: Tensor,
                           agent_identity: int,
                           x_min: float,
                           x_max: float,
                           y_min: float,
                           y_max: float,
                           goal_line: float,
                           variables_per_agent_per_timestep: int,
                           backward_view: int,
                           rows: int,
                           cols: int,
                           window_rows: int,
                           window_cols: int,
                           device: str,
                           person_radius: float,
                           standard_deviation: float,
                           influence_radius: float,
                           runner_identities: List[int],
                           start_line: float,
                           **kwargs):
    entrance_line = y_min - start_line
    y_min = 0
    y_max = y_max - start_line
    goal_line = goal_line - start_line

    # Render scenario
    new_state = state.clone()
    for a in range(new_state.shape[0]):
        new_state[a, 1] = new_state[a, 1] - start_line
        new_state[a, 1 + 4] = new_state[a, 1 + 4] - start_line
        new_state[a, 1 + 2 * 4] = new_state[a, 1 + 2 * 4] - start_line
    del state
    number_persons = new_state.shape[0]
    color_wall = torch.tensor([0, 0, 1], device=device).float()
    color_goal = torch.tensor([0, 1, 0], device=device).float()
    color_others = torch.tensor([1, 0, 0], device=device).float()
    color_current = torch.tensor([1, 1, 1], device=device).float()
    color_reference_line = torch.tensor([0.5, 0.5, 0.5], device=device).float()

    row_radius, col_radius = torch.tensor((window_rows - 1.) / 2, device=device).int(), \
                             torch.tensor((window_cols - 1.) / 2, device=device).int()
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    # + 40 / + 120 because the runner has negative y-coordinates in his initial positions
    extended_scenario_array = 0.2 * torch.ones([3, ext_scen_size_row + 120, ext_scen_size_col])
    agent_windows = torch.zeros([3 * backward_view, window_rows, window_cols], device=device)

    for i in range(backward_view):
        scenario_array = scenario_torch(
            new_state[:, i * variables_per_agent_per_timestep: i * variables_per_agent_per_timestep + 2],
            agent_identity, color_current, color_others, color_wall, color_goal, person_radius,
            x_min, x_max, y_min, y_max, goal_line, entrance_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, device
        )

        # Coordinates and movement of agent
        u, v = cart_to_img_torch(
            new_state[agent_identity, i * variables_per_agent_per_timestep:
                                      i * variables_per_agent_per_timestep + 2],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        looking_in_cartesian_direction = new_state[agent_identity, i * variables_per_agent_per_timestep:
                                                                   i * variables_per_agent_per_timestep + 2] + \
                                         new_state[agent_identity, i * variables_per_agent_per_timestep + 2:
                                                                   i * variables_per_agent_per_timestep + 4]

        # Add reference lines
        if agent_identity in runner_identities:
            scenario_array[:u, v] += color_reference_line

        # plt.imshow(scenario_array.cpu().numpy());
        # plt.show()

        # Fill into extended scenario
        extended_scenario_array[:, row_radius:ext_scen_size_row - row_radius,
        col_radius:ext_scen_size_col - col_radius] = scenario_array.permute(2, 0, 1)

        # Rotate
        extended_scenario_array_rotated = rotate_image(
            extended_scenario_array,
            angle=0,
            center_u=u,
            center_v=v + col_radius
        )

        # plt.imshow(extended_scenario_array_rotated.cpu().numpy().transpose(1,2,0));
        # plt.show()

        # Cut window around agent
        # agent_window = extended_scenario_array_rotated[:, u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1)]
        agent_window = cut_window(extended_scenario_array_rotated, window_rows, window_cols)
        agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window

        # plt.imshow(agent_window.cpu().numpy().transpose(1,2,0));
        # plt.show()
        #
        # print('OK')

    agent_windows = agent_windows.unsqueeze(0)
    # vis_feature_maps(agent_windows)

    # Clean up
    del color_wall, color_goal, color_others, color_current, row_radius, col_radius, scenario_array, u, v  # , fov_radius
    del extended_scenario_array, agent_window

    return agent_windows


def render_vision_field_torch(state: Tensor,
                              agent_identity: int,
                              x_min: float,
                              x_max: float,
                              y_min: float,
                              y_max: float,
                              goal_line: float,
                              variables_per_agent_per_timestep: int,
                              backward_view: int,
                              rows: int,
                              cols: int,
                              window_rows: int,
                              window_cols: int,
                              device: str,
                              person_radius: float,
                              standard_deviation: float,
                              influence_radius: float,
                              runner_identities: List[int],
                              start_line: float,
                              **kwargs):
    entrance_line = y_min - start_line
    y_min = 0
    y_max = y_max - start_line
    goal_line = goal_line - start_line

    # Render scenario
    new_state = state.clone()
    for a in range(new_state.shape[0]):
        new_state[a, 1] = new_state[a, 1] - start_line
        new_state[a, 1 + 4] = new_state[a, 1 + 4] - start_line
        new_state[a, 1 + 2 * 4] = new_state[a, 1 + 2 * 4] - start_line
    del state
    number_persons = new_state.shape[0]
    color_wall = torch.tensor([0, 0, 1], device=device).float()
    color_goal = torch.tensor([0, 1, 0], device=device).float()
    color_others = torch.tensor([1, 0, 0], device=device).float()
    color_current = torch.tensor([1, 1, 1], device=device).float()
    color_reference_line = torch.tensor([0.5, 0.5, 0.5], device=device).float()

    row_radius, col_radius = torch.tensor((window_rows - 1.) / 2, device=device).int(), \
                             torch.tensor((window_cols - 1.) / 2, device=device).int()
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    # + 40 / + 120 because the runner has negative y-coordinates in his initial positions
    extended_scenario_array = 0.2 * torch.ones([3, ext_scen_size_row + 120, ext_scen_size_col])
    agent_windows = torch.zeros([3 * backward_view, window_rows, window_cols], device=device)

    for i in range(backward_view):
        scenario_array = scenario_torch(
            new_state[:, i * variables_per_agent_per_timestep: i * variables_per_agent_per_timestep + 2],
            agent_identity, color_current, color_others, color_wall, color_goal, person_radius,
            x_min, x_max, y_min, y_max, goal_line, entrance_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, device
        )

        # Coordinates and movement of agent
        u, v = cart_to_img_torch(
            new_state[agent_identity, i * variables_per_agent_per_timestep:
                                      i * variables_per_agent_per_timestep + 2],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        looking_in_cartesian_direction = new_state[agent_identity, i * variables_per_agent_per_timestep:
                                                                   i * variables_per_agent_per_timestep + 2] + \
                                         new_state[agent_identity, i * variables_per_agent_per_timestep + 2:
                                                                   i * variables_per_agent_per_timestep + 4]
        u_lookdirimg, v_lookdirimg = cart_to_img_torch(
            looking_in_cartesian_direction, x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        du, dv = u_lookdirimg - u, v_lookdirimg - v

        # To ensure that the agent circle is included
        if x_max - x_min > y_max - y_min:
            person_radius_px = torch.tensor(rows * person_radius * 1. / (y_max - y_min), device=device).int()
        else:
            person_radius_px = torch.tensor(cols * person_radius * 1. / (x_max - x_min), device=device).int()
        shift_length = torch.sqrt(du.float() ** 2 + dv.float() ** 2)
        if shift_length == 0.:
            shift_length = shift_length + person_radius_px
        normalized_du, normalized_dv = du.float() / shift_length, dv.float() / shift_length
        if normalized_du > 0:
            normalized_du = normalized_du.ceil()
        else:
            normalized_du = normalized_du.floor()
        if normalized_dv > 0:
            normalized_dv = normalized_dv.ceil()
        else:
            normalized_dv = normalized_dv.floor()

        # Mask field of view
        fov_radius = row_radius if window_rows < window_cols else col_radius
        scenario_array = scenario_array * (
            create_circular_mask_torch(rows, cols, u - normalized_du,
                                       v - normalized_dv, fov_radius).unsqueeze(2).repeat(1, 1, 3))
        scenario_array = scenario_array + (
                0.2 * create_inverse_circular_mask_torch(rows, cols, u - normalized_du,
                                                         v - normalized_dv, fov_radius).unsqueeze(2).repeat(1, 1, 3))

        # Mask orientation
        scenario_array = scenario_array * (
            create_angular_mask_torch(rows, cols, u - normalized_du,
                                      v - normalized_dv, du.float(), dv.float(), math_pi / 2.).unsqueeze(2).repeat(1, 1,
                                                                                                                   3))
        scenario_array = scenario_array + (
                0.2 * create_inverse_angular_mask_torch(rows, cols, u - normalized_du,
                                                        v - normalized_dv, du.float(), dv.float(),
                                                        math_pi / 2.).unsqueeze(2).repeat(1, 1, 3))

        # Add reference lines
        if agent_identity in runner_identities:
            scenario_array[:u, v] += color_reference_line
        else:
            goal = new_state[agent_identity, 2 * 4: 2 * 4 + 2]  # second last position
            u_goal, v_goal = cart_to_img_torch(goal, x_min, x_max, y_min, y_max, rows, cols, person_radius)
            v_start, v_end = v, v_goal
            u_start, u_end = u, u_goal
            start = torch.cat([u_start.unsqueeze(0), v_start.unsqueeze(0)]).float()
            end = torch.cat([u_end.unsqueeze(0), v_end.unsqueeze(0)]).float()
            dir = end - start
            length = torch.norm(dir).ceil()
            steps = torch.arange(length).int().to(device)
            ts = steps.float() / length
            line = torch.zeros(length.int(), 2, device=device)
            for t, step in zip(ts, steps):
                line[step, :] = start + t * dir
            line = line.round().long()
            line_rows = line[:, 0]
            line_columns = line[:, 1]
            scenario_array[line_rows, line_columns] += color_reference_line

        # plt.imshow(scenario_array.cpu().numpy());
        # plt.show()

        # Fill into extended scenario
        extended_scenario_array[:, row_radius:ext_scen_size_row - row_radius,
        col_radius:ext_scen_size_col - col_radius] = scenario_array.permute(2, 0, 1)

        # Rotate
        angle = angle_2D_full(  # angle of orientation in [0, 2pi]
            new_state[agent_identity, i * variables_per_agent_per_timestep + 2:
                                      i * variables_per_agent_per_timestep + 4],
            torch.tensor([0., -1.], device=device)  # in row direction (negative y-axis -> positive row-axis)
        )
        extended_scenario_array_rotated = rotate_image(extended_scenario_array, 2 * math_pi - angle,
                                                       center_u=u + row_radius,
                                                       center_v=v + col_radius)

        # Cut window around agent
        # agent_window = extended_scenario_array_rotated[:, u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1)]
        agent_window = cut_window(extended_scenario_array_rotated, window_rows, window_cols)
        agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window

    agent_windows = agent_windows.unsqueeze(0)
    # vis_feature_maps(agent_windows)

    # Clean up
    del color_wall, color_goal, color_others, color_current, row_radius, col_radius, scenario_array, u, v  # , fov_radius
    del extended_scenario_array, agent_window

    return agent_windows


def scenario_torch(positions: torch.Tensor, agent_identity: int,
                   color_current: torch.Tensor, color_others: torch.Tensor, color_obstacle: torch.Tensor,
                   color_goal_line: torch.Tensor,
                   person_radius: float, x_min: float, x_max: float, y_min: float, y_max: float,
                   goal_line, entrance_line, variables_per_agent_per_timestep: int, backward_view: int, rows: int,
                   cols: int,
                   device: str) -> torch.Tensor:
    # Setup coordinates
    ## Scene
    tl_xy = torch.tensor([x_min, y_max], device=device)
    br_xy = torch.tensor([x_max, y_min], device=device)
    tl_u, tl_v = cart_to_img_torch(tl_xy, x_min, x_max, y_min, y_max, rows, cols, person_radius)
    br_u, br_v = cart_to_img_torch(br_xy, x_min, x_max, y_min, y_max, rows, cols, person_radius)
    goal_line_u = torch.tensor(rows * (1 - goal_line * 1. / (y_max - y_min)), device=device).int()  # only row necessary
    entrance_line_u = torch.tensor(rows * (1 - entrance_line * 1. / (y_max - y_min)), device=device).int()

    ## Agents & their masks
    # Choose smaller dimension for radius
    if x_max - x_min > y_max - y_min:
        person_radius_px = torch.tensor(rows * person_radius * 1. / (y_max - y_min), device=device).int()
    else:
        person_radius_px = torch.tensor(cols * person_radius * 1. / (x_max - x_min), device=device).int()
    number_agents = positions.shape[0]
    agents_circular_masks = torch.zeros([number_agents, rows, cols], device=device)
    for i in range(number_agents):
        u, v = cart_to_img_torch(positions[i], x_min, x_max, y_min, y_max, rows, cols, person_radius)
        agents_circular_masks[i] = create_circular_mask_torch(rows, cols, u, v, person_radius_px)

    # Create scenario
    scenario = torch.zeros([rows, cols, 3], device=device)

    ## Walls
    scenario[tl_u: br_u, tl_v, :] += color_obstacle
    scenario[tl_u: br_u, br_v - 1, :] += color_obstacle

    ## Goal & entrance lines
    scenario[goal_line_u, tl_v + 1: br_v - 1, :] += color_goal_line
    scenario[entrance_line_u, tl_v + 1: br_v - 1, :] += color_goal_line * 0.7

    ## Agents
    agents_masks = torch.zeros([number_agents, rows, cols, 3], device=device)
    for i in range(number_agents):
        color = color_current if i == agent_identity else color_others
        color = color.unsqueeze(0).unsqueeze(0)
        scenario += agents_circular_masks[i].unsqueeze(2).repeat(1, 1, 3).float() * color

    # Clean up
    del tl_xy, br_xy, tl_u, tl_v, br_u, br_v, goal_line, person_radius_px, agents_circular_masks, color, agents_masks

    return scenario


def render_density_field(state: Tensor,
                         agent_identity: int,
                         x_min: float,
                         x_max: float,
                         y_min: float,
                         y_max: float,
                         variables_per_agent_per_timestep: int,
                         backward_view: int,
                         rows: int,
                         cols: int,
                         window_rows: int,
                         window_cols: int,
                         device: str,
                         person_radius: float,
                         standard_deviation: float,
                         influence_radius: float,
                         **kwargs):
    # Render scenario
    state_clone = state.clone()

    row_radius, col_radius = int((window_rows - 1.) / 2), int((window_cols - 1.) / 2)
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    extended_scenario_array = numpy.zeros([ext_scen_size_row, ext_scen_size_col])
    agent_windows = numpy.zeros([backward_view, window_rows, window_cols])
    gaussian_density = render_gaussian_density(
        state, x_min, x_max, y_min, y_max, variables_per_agent_per_timestep, backward_view,
        rows, cols, device, person_radius, standard_deviation, influence_radius
    ).cpu().numpy()
    for i in range(backward_view):
        scenario_array = gaussian_density[0, i]
        scenario_array = (scenario_array * 1. / scenario_array.max())

        # Coordinates of agent
        u, v = cart_to_img(
            state_clone[agent_identity, i * variables_per_agent_per_timestep],
            state_clone[agent_identity, i * variables_per_agent_per_timestep + 1],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        u, v = int(u.cpu().numpy()), int(v.cpu().numpy())

        # Mask field of view
        movement_goal = state_clone[agent_identity, :2] + state[agent_identity, 2:4]
        goal_u, goal_v = cart_to_img(movement_goal[0], movement_goal[1],
                                     x_min, x_max, y_min, y_max, rows, cols,
                                     person_radius)
        goal_shift = numpy.array([goal_u.cpu().numpy() - u, goal_v.cpu().numpy() - v])
        fov_radius = row_radius if window_rows < window_cols else col_radius
        for r in range(scenario_array.shape[0]):
            for c in range(scenario_array.shape[1]):
                shift = numpy.array([r - u, c - v])
                if shift[0] ** 2 + shift[1] ** 2 > fov_radius ** 2:  # pixel outside circle around agent
                    scenario_array[r, c] = 0.5
                if numpy.dot(shift, goal_shift) < 0:
                    scenario_array[r, c] = 0.5

        # Fill into extended scenario
        extended_scenario_array[
        row_radius:ext_scen_size_row - row_radius,
        col_radius:ext_scen_size_col - col_radius
        ] = scenario_array

        # Cut window around agent
        agent_window = extended_scenario_array[u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1)]

        agent_windows[i: i + 1] = agent_window

    agent_windows = torch.tensor(agent_windows, device=device).float().unsqueeze(0)
    return agent_windows


def render_scenario_fast(state: Tensor,
                         agent_identity: int,
                         x_min: float,
                         x_max: float,
                         y_min: float,
                         y_max: float,
                         goal_line: float,
                         variables_per_agent_per_timestep: int,
                         backward_view: int,
                         rows: int,
                         cols: int,
                         window_rows: int,
                         window_cols: int,
                         device: str,
                         person_radius: float,
                         standard_deviation: float,
                         influence_radius: float,
                         **kwargs):
    # Render scenario
    number_persons = state.shape[0]
    other_person_slots = numpy.array([0, 1, 0], dtype=numpy.float16)
    obstacle_slots = numpy.array([1, 1, 1], dtype=numpy.float16)
    goal_line_slots = numpy.array([0, 0, 1], dtype=numpy.float16)
    own_slots = numpy.array([1, 0, 0], dtype=numpy.float16)

    row_radius, col_radius = int((window_rows - 1) / 2), int((window_cols - 1) / 2)
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    extended_scenario_array = numpy.zeros([ext_scen_size_row, ext_scen_size_col, 3])
    agent_windows = numpy.zeros([3 * backward_view, window_rows, window_cols])
    for i in range(backward_view):

        scenario_array = numpy.zeros([3, rows, cols])

        # Other pedestrians
        for a in range(number_persons):
            if a == agent_identity:
                slots = own_slots
            else:
                slots = other_person_slots
            u, v = cart_to_img(state[a, i * variables_per_agent_per_timestep], state[agent_identity, 1],
                               x_min, x_max, y_min, y_max, rows, cols, person_radius)
            u, v = int(u.cpu().numpy()), int(v.cpu().numpy())
            scenario_array[:, u, v] = slots

        # Walls
        vertical_obstacle = obstacle_slots.repeat(rows).reshape(3, rows)
        scenario_array[:, :, 0] = vertical_obstacle
        scenario_array[:, :, cols - 1] = vertical_obstacle

        # Goal line
        u, _ = cart_to_img(0, goal_line,
                           x_min, x_max, y_min, y_max, rows, cols, person_radius)
        u = int(u.cpu().numpy())
        scenario_array[:, u, 1: cols - 2] = goal_line_slots.repeat(3, cols - 2)

        extended_scenario_array[row_radius:ext_scen_size_row - row_radius, col_radius:ext_scen_size_col - col_radius,
        :] = scenario_array

        # Cut window around agent
        u, v = cart_to_img(state[agent_identity, i * variables_per_agent_per_timestep], state[agent_identity, 1],
                           x_min, x_max, y_min, y_max, rows, cols, person_radius)
        u, v = int(u.cpu().numpy()), int(v.cpu().numpy())
        agent_window = extended_scenario_array[u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1), :]
        try:
            agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window.transpose(2, 0, 1)
        except:
            print('FAIL')

    agent_windows = torch.tensor(agent_windows, device=device).float().unsqueeze(0)
    return agent_windows


def render_vision_field_torch_reduced(state: Tensor,
                                      agent_identity: int,
                                      x_min: float,
                                      x_max: float,
                                      y_min: float,
                                      y_max: float,
                                      goal_line: float,
                                      variables_per_agent_per_timestep: int,
                                      backward_view: int,
                                      rows: int,
                                      cols: int,
                                      window_rows: int,
                                      window_cols: int,
                                      device: str,
                                      person_radius: float,
                                      standard_deviation: float,
                                      influence_radius: float,
                                      runner_identities: List[int],
                                      **kwargs):
    # Render scenario
    new_state = state.clone()
    del state
    number_persons = new_state.shape[0]
    color_wall = torch.tensor([0, 0, 1], device=device).float()
    color_goal = torch.tensor([0, 1, 0], device=device).float()
    color_others = torch.tensor([1, 0, 0], device=device).float()
    color_current = torch.tensor([1, 1, 1], device=device).float()
    color_reference_line = torch.tensor([0.5, 0.5, 0.5], device=device).float()

    row_radius, col_radius = torch.tensor((window_rows - 1.) / 2, device=device).int(), \
                             torch.tensor((window_cols - 1.) / 2, device=device).int()
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    # + 40 because the runner has negative y-coordinates in his initial positions
    extended_scenario_array = torch.zeros([ext_scen_size_row + 80, ext_scen_size_col, 3])
    agent_windows = torch.zeros([3 * backward_view, window_rows, window_cols], device=device)

    for i in range(backward_view):
        scenario_array = scenario_torch(
            new_state[:, i * variables_per_agent_per_timestep: i * variables_per_agent_per_timestep + 2],
            agent_identity, color_current, color_others, color_wall, color_goal, person_radius,
            x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, device
        )

        # Coordinates and movement of agent
        u, v = cart_to_img_torch(
            new_state[agent_identity, i * variables_per_agent_per_timestep:
                                      i * variables_per_agent_per_timestep + 2],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        # u0, v0 = cart_to_img_torch(
        #     new_state[agent_identity, (i + 1) * variables_per_agent_per_timestep:
        #                               (i + 1) * variables_per_agent_per_timestep + 2],
        #     x_min, x_max, y_min, y_max, rows, cols, person_radius
        # )
        # du, dv = u - u0, v - v0

        # Add reference line for runners
        if agent_identity in runner_identities:
            scenario_array[:u, v] += color_reference_line

        # Mask field of view
        fov_radius = row_radius if window_rows < window_cols else col_radius
        scenario_array = scenario_array * (
            create_circular_mask_torch(rows, cols, u, v, fov_radius).unsqueeze(2).repeat(1, 1, 3))

        # scenario_array = scenario_array * (
        #     create_angular_mask_torch(rows, cols, u, v, du.float(), dv.float(), math_pi / 2.).unsqueeze(2).repeat(1, 1,
        #                                                                                                           3))
        # plt.imshow(scenario_array.cpu().numpy());
        # plt.show()

        # Fill into extended scenario
        extended_scenario_array[row_radius:ext_scen_size_row - row_radius, col_radius:ext_scen_size_col - col_radius,
        :] = scenario_array

        # Cut window around agent
        agent_window = extended_scenario_array[u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1),
                       :]

        agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window.permute(2, 0, 1)
    agent_windows = agent_windows.unsqueeze(0)
    # vis_feature_maps(agent_windows)

    # Clean up
    del color_wall, color_goal, color_others, color_current, row_radius, col_radius, scenario_array, u, v  # , fov_radius
    del extended_scenario_array, agent_window

    return agent_windows


def render_vision_field_torch_reduced(state: Tensor,
                                      agent_identity: int,
                                      x_min: float,
                                      x_max: float,
                                      y_min: float,
                                      y_max: float,
                                      goal_line: float,
                                      variables_per_agent_per_timestep: int,
                                      backward_view: int,
                                      rows: int,
                                      cols: int,
                                      window_rows: int,
                                      window_cols: int,
                                      device: str,
                                      person_radius: float,
                                      standard_deviation: float,
                                      influence_radius: float,
                                      runner_identities: List[int],
                                      **kwargs):
    # Render scenario
    new_state = state.clone()
    del state
    number_persons = new_state.shape[0]
    color_wall = torch.tensor([0, 0, 1], device=device).float()
    color_goal = torch.tensor([0, 1, 0], device=device).float()
    color_others = torch.tensor([1, 0, 0], device=device).float()
    color_current = torch.tensor([1, 1, 1], device=device).float()
    color_reference_line = torch.tensor([0.5, 0.5, 0.5], device=device).float()

    row_radius, col_radius = torch.tensor((window_rows - 1.) / 2, device=device).int(), \
                             torch.tensor((window_cols - 1.) / 2, device=device).int()
    ext_scen_size_row, ext_scen_size_col = rows + 2 * row_radius, cols + 2 * col_radius
    # + 40 because the runner has negative y-coordinates in his initial positions
    extended_scenario_array = torch.zeros([ext_scen_size_row + 80, ext_scen_size_col, 3])
    agent_windows = torch.zeros([3 * backward_view, window_rows, window_cols], device=device)

    for i in range(backward_view):
        scenario_array = scenario_torch(
            new_state[:, i * variables_per_agent_per_timestep: i * variables_per_agent_per_timestep + 2],
            agent_identity, color_current, color_others, color_wall, color_goal, person_radius,
            x_min, x_max, y_min, y_max, goal_line,
            variables_per_agent_per_timestep, backward_view, rows, cols, device
        )

        # Coordinates and movement of agent
        u, v = cart_to_img_torch(
            new_state[agent_identity, i * variables_per_agent_per_timestep:
                                      i * variables_per_agent_per_timestep + 2],
            x_min, x_max, y_min, y_max, rows, cols, person_radius
        )
        # u0, v0 = cart_to_img_torch(
        #     new_state[agent_identity, (i + 1) * variables_per_agent_per_timestep:
        #                               (i + 1) * variables_per_agent_per_timestep + 2],
        #     x_min, x_max, y_min, y_max, rows, cols, person_radius
        # )
        # du, dv = u - u0, v - v0

        # Add reference line for runners
        if agent_identity in runner_identities:
            scenario_array[:u, v] += color_reference_line

        # Mask field of view
        fov_radius = row_radius if window_rows < window_cols else col_radius
        scenario_array = scenario_array * (
            create_circular_mask_torch(rows, cols, u, v, fov_radius).unsqueeze(2).repeat(1, 1, 3))

        # scenario_array = scenario_array * (
        #     create_angular_mask_torch(rows, cols, u, v, du.float(), dv.float(), math_pi / 2.).unsqueeze(2).repeat(1, 1,
        #                                                                                                           3))
        # plt.imshow(scenario_array.cpu().numpy());
        # plt.show()

        # Fill into extended scenario
        extended_scenario_array[row_radius:ext_scen_size_row - row_radius, col_radius:ext_scen_size_col - col_radius,
        :] = scenario_array

        # Cut window around agent
        agent_window = extended_scenario_array[u:(u + 2 * row_radius + 1), v:(v + 2 * col_radius + 1),
                       :]

        agent_windows[3 * i: 3 * (i + 1), :, :] = agent_window.permute(2, 0, 1)
    agent_windows = agent_windows.unsqueeze(0)
    # vis_feature_maps(agent_windows)

    # Clean up
    del color_wall, color_goal, color_others, color_current, row_radius, col_radius, scenario_array, u, v  # , fov_radius
    del extended_scenario_array, agent_window

    return agent_windows


def generate_kinematics_torch(state: Tensor,
                              agent_identity: int,
                              x_min: float,
                              x_max: float,
                              y_min: float,
                              y_max: float,
                              goal_line: float,
                              runner_identities: List[int],
                              device: str,
                              soft_person_radius: float,
                              time_per_step: float,
                              start_line: float,
                              **kwargs):
    new_state = state.clone()
    del state

    # Initialization of kinematics
    n_agents = new_state.shape[0]
    kinematics = torch.zeros(n_agents * 3 + 6 * 2, device=device)  # self, others, walls

    # Auxiliary
    p = new_state[agent_identity, :2]
    v = (p - new_state[agent_identity, 4:6])
    direction = new_state[agent_identity, 2:4]
    if agent_identity in runner_identities:
        goal = torch.tensor([new_state[agent_identity, 0], goal_line], device=device).float()
    else:
        goal = torch.tensor([0., 0.], device=device).float()
    shift_goal = goal - p
    a = angle_2D_full(new_state[agent_identity, 2:4], shift_goal)
    other_ids = []
    for id in range(n_agents):
        if id != agent_identity:
            other_ids += [id]
    p_agents = new_state[other_ids, :2]
    shifts_agents = p_agents - p
    v_agents = new_state[other_ids, :2] - new_state[other_ids, 4:6]
    a_agents = new_state[other_ids, 3]

    # Agent
    # S_ag = torch.norm(v)
    # angle_v_to_reference_line = angle_2D_full(v, shift_goal)
    S_ag = torch.norm(v) # * torch.cos(angle_v_to_reference_line)
    D_ag = torch.norm(shift_goal)
    A_v = angle_2D_full(direction, shift_goal)
    kinematics[0] = S_ag
    kinematics[1] = A_v
    kinematics[2] = D_ag

    # Other agents
    dagis = torch.zeros(n_agents - 1)
    for i in range(len(other_ids)):
        D_ag_i = torch.norm(shifts_agents[i]) - 2 * soft_person_radius
        dagis[i] = D_ag_i
    sorted_args = torch.argsort(dagis)

    for i, id in enumerate(sorted_args):
        angle_v_to_connection_line = angle_2D_full(v, shifts_agents[id])
        angle_vi_to_connection_line = angle_2D_full(v_agents[id], shifts_agents[id])
        proj_v_to_connection_line = torch.norm(v) * torch.cos(angle_v_to_connection_line)
        proj_vi_to_connection_line = torch.norm(v_agents[id]) * torch.cos(angle_vi_to_connection_line)
        S_rel_id = proj_v_to_connection_line + proj_vi_to_connection_line
        A_ag_id = angle_2D_full(shifts_agents[id], shift_goal)
        kinematics[(i + 1) * 3] = S_rel_id
        kinematics[(i + 1) * 3 + 1] = dagis[id]
        kinematics[(i + 1) * 3 + 2] = A_ag_id

    # Walls
    shift_x_min = torch.tensor([x_min - p[0], 0.], device=device)
    shift_x_max = torch.tensor([x_max - p[0], 0.], device=device)
    shift_y_min = torch.tensor([0., y_min - p[1]], device=device)
    shift_y_max = torch.tensor([0., y_max - p[1]], device=device)
    shift_goal_line = torch.tensor([0., goal_line - p[1]], device=device)
    shift_start_line = torch.tensor([0., start_line - p[1]], device=device)
    d_x_min = torch.norm(shift_x_min).unsqueeze(0)
    d_x_max = torch.norm(shift_x_max).unsqueeze(0)
    d_y_min = torch.norm(shift_y_min).unsqueeze(0)
    d_y_max = torch.norm(shift_y_max).unsqueeze(0)
    d_goal_line = torch.norm(shift_goal_line).unsqueeze(0)
    d_start_line = torch.norm(shift_start_line).unsqueeze(0)
    a_x_min = angle_2D_full(shift_x_min, shift_goal).unsqueeze(0)
    a_x_max = angle_2D_full(shift_x_max, shift_goal).unsqueeze(0)
    a_y_min = angle_2D_full(shift_y_min, shift_goal).unsqueeze(0)
    a_y_max = angle_2D_full(shift_y_max, shift_goal).unsqueeze(0)
    a_goal_line = angle_2D_full(shift_goal_line, shift_goal).unsqueeze(0)
    a_start_line = angle_2D_full(shift_start_line, shift_goal).unsqueeze(0)

    distances = torch.cat([d_x_min, d_x_max, d_y_min, d_y_max, d_start_line, d_goal_line])
    angles = torch.cat([a_x_min, a_x_max, a_y_min, a_y_max, a_start_line, a_goal_line])
    #sorted_args = torch.argsort(distances)
    #for i, id in enumerate(sorted_args):
    for i in range(6):
        kinematics[3 * n_agents + 2 * i] = distances[i] - soft_person_radius
        kinematics[3 * n_agents + 2 * i + 1] = angles[i]

    break_if_nan(kinematics)
    return kinematics.unsqueeze(0)


def generate_kinematics_torch2(state: Tensor,
                               agent_identity: int,
                               x_min: float,
                               x_max: float,
                               y_min: float,
                               y_max: float,
                               goal_line: float,
                               runner_identities: List[int],
                               device: str,
                               soft_person_radius: float,
                               time_per_step: float,
                               start_line: float,
                               **kwargs):
    new_state = state.clone()
    del state
    right_axis = torch.tensor([1., 0.], device=new_state.device)
    origin = torch.tensor([0., 0.], device=new_state.device)

    # Initialization of kinematics
    n_agents = new_state.shape[0]
    kinematics = torch.zeros(n_agents * 4 + 6 * 2, device=device)  # self, others, walls

    # Auxiliary
    p = new_state[agent_identity, :2]
    v = (p - new_state[agent_identity, 4:6])
    direction = torch.cat([new_state[agent_identity, 2:3], new_state[agent_identity, 3:4]])
    a = angle_2D_full(new_state[agent_identity, 2:4], right_axis)
    other_ids = []
    for id in range(n_agents):
        if id != agent_identity:
            other_ids += [id]
    p_agents = new_state[other_ids, :2]
    shifts_agents = p_agents - p
    v_agents = new_state[other_ids, :2] - new_state[other_ids, 4:6]
    a_agents = new_state[other_ids, 3]
    rel_v = v - v_agents

    # Agent
    D_ag = torch.norm(p)
    A_ag = angle_2D_full(p, right_axis)
    S_ag = torch.norm(v)
    A_v = angle_2D_full(direction, right_axis)
    kinematics[0] = D_ag
    kinematics[1] = A_ag
    kinematics[2] = S_ag
    kinematics[3] = A_v

    # Other agents
    dagis = torch.zeros(n_agents - 1)
    for i in range(len(other_ids)):
        D_ag_i = torch.norm(shifts_agents[i]) - 2 * soft_person_radius
        dagis[i] = D_ag_i
    sorted_args = torch.argsort(dagis)

    for i, id in enumerate(sorted_args):
        D_ag_id = torch.norm(p_agents[id])
        A_ag_id = angle_2D_full(p_agents[id], right_axis)
        S_id = torch.norm(v_agents[id])
        A_id = angle_2D_full(v_agents[id], right_axis)
        kinematics[(i + 1) * 4] = D_ag_id
        kinematics[(i + 1) * 4 + 1] = A_ag_id
        kinematics[(i + 1) * 4 + 2] = S_id
        kinematics[(i + 1) * 4 + 3] = A_id

    # Walls
    shift_x_min = torch.tensor([x_min - p[0], 0.], device=device)
    shift_x_max = torch.tensor([x_max - p[0], 0.], device=device)
    shift_y_min = torch.tensor([0., y_min - p[1]], device=device)
    shift_y_max = torch.tensor([0., y_max - p[1]], device=device)
    shift_goal_line = torch.tensor([0., goal_line - p[1]], device=device)
    shift_start_line = torch.tensor([0., start_line - p[1]], device=device)
    d_x_min = torch.norm(shift_x_min).unsqueeze(0)
    d_x_max = torch.norm(shift_x_max).unsqueeze(0)
    d_y_min = torch.norm(shift_y_min).unsqueeze(0)
    d_y_max = torch.norm(shift_y_max).unsqueeze(0)
    d_goal_line = torch.norm(shift_goal_line).unsqueeze(0)
    d_start_line = torch.norm(shift_start_line).unsqueeze(0)
    # d_x_min = torch.tensor([x_min - p[0]])
    # d_x_max = torch.tensor([x_max - p[0]])
    # d_y_min = torch.tensor([y_min - p[1]])
    # d_y_max = torch.tensor([y_max - p[1]])
    # d_goal_line = torch.tensor([goal_line - p[1]])
    # d_start_line = torch.tensor([start_line - p[1]])
    a_x_min = angle_2D_full(shift_x_min, right_axis).unsqueeze(0)
    a_x_max = angle_2D_full(shift_x_max, right_axis).unsqueeze(0)
    a_y_min = angle_2D_full(shift_y_min, right_axis).unsqueeze(0)
    a_y_max = angle_2D_full(shift_y_max, right_axis).unsqueeze(0)
    a_goal_line = angle_2D_full(shift_goal_line, right_axis).unsqueeze(0)
    a_start_line = angle_2D_full(shift_start_line, right_axis).unsqueeze(0)

    distances = torch.cat([d_x_min, d_x_max, d_y_min, d_y_max, d_start_line, d_goal_line])
    angles = torch.cat([a_x_min, a_x_max, a_y_min, a_y_max, a_start_line, a_goal_line])
    sorted_args = torch.argsort(distances)
    for i, id in enumerate(sorted_args):
        kinematics[n_agents * 4 + 2 * i] = distances[id] - soft_person_radius
        kinematics[n_agents * 4 + 2 * i + 1] = angles[id]

    break_if_nan(kinematics)
    return kinematics.unsqueeze(0)
