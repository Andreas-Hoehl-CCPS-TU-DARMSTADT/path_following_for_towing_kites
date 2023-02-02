"""calculates the optimal trajectory of the Erhard prediction_model.

The optimal Trajectory for a given parameter set of a towing kite prediction_model (Erhard) is calculated
using the IPOPT solver of casadi.
The user can change the option settings in order to calculate a trajectory with the desired configuration.
"""
import casadi as ca
import matplotlib.pyplot as plt

from visualisation import trajectory_plotting
import numpy as np
import model_zoo
import utils
import time
from datetime import datetime
from interfaces import ResidualModel
from scipy.interpolate import CubicSpline

# ______ options ______
_TRAJECTORY_ID = f'test'  # name of the trajectory
_COMMENT = 'hand created'  # additional comment that is stored in the trajectory doc
_KITE_PARAMETER = model_zoo.ERHARD_PHYSICAL_PARAMETER  # model parameter
_RESIDUAL_MODEL_NAME = 'default_mlp'  # name of the residual model e.g. 'default_mlp'
_RESIDUAL_THRUST_MODEL_NAME = 'default_mlp'  # name of the residual model e.g. 'default_mlp'
_CALCULATE_HALF = False  # calculate only half of the trajectory and get the other half by symmetry
_USE_STRONGER_CONSTRAINTS = True  # stronger constraint on theta
_FIX_TS = True  # fix the sampling time. If false the final time is an optimisation variable
_USE_INIT = True  # uses free Ts physical trajectory as initial guess
_LIMIT_DELTA_U = True
_T_OFFSET = 0
_N0 = 71


def _get_initial_guess(n, use_initial_guess, calculate_half):
    if use_initial_guess:
        trajectory = utils.load_optimal_trajectories('physical_trajectory_free_Ts')
        t_final_init = trajectory.t[-1]
        spline_x = CubicSpline(trajectory.t, trajectory.x, bc_type='periodic')
        spline_u = CubicSpline(trajectory.t, np.concatenate((trajectory.u.flatten(), trajectory.u[0])).reshape(-1, 1),
                               bc_type='periodic')
        if calculate_half:
            time_vector = np.linspace(t_final_init / n, t_final_init / 2, n + 1)
        else:
            time_vector = np.linspace(t_final_init / n, t_final_init, n + 1)

        initial_state_trajectory = spline_x(time_vector).tolist()
        initial_input_trajectory = spline_u(time_vector).tolist()[:-1]
    else:
        # guess for initial state
        theta_init = 30 * np.pi / 180  # in [rad]
        phi_init = 0 * np.pi / 180  # in [rad]
        psi_init = -10 * np.pi / 180  # in [rad]
        x_init = [theta_init, phi_init, psi_init]

        initial_state_trajectory = [x_init] + [[0.5, 0, 0]] * n
        initial_input_trajectory = [[0]] * n
        t_final_init = 20

    return initial_state_trajectory, initial_input_trajectory, t_final_init


def _calculate_optimal_trajectory(n, kite_parameter, fix_ts, comment, calculate_half,
                                  residual_state_model: ResidualModel, residual_thrust_model: ResidualModel,
                                  use_stronger_constraints, use_initial_guess, number_of_steps, limit_delta_u):
    if (residual_state_model is not None or residual_thrust_model is not None) and not fix_ts:
        raise RuntimeError('residual model can only be used in combination with a fixed sampling time')
    # step size
    if not fix_ts:
        t_final = ca.MX.sym('t_final', 1)
        h = t_final / n
    else:
        h = model_zoo.ERHARD_Ts
        t_final = n * h

    initial_state_trajectory, initial_input_trajectory, t_final_init = _get_initial_guess(n, use_initial_guess,
                                                                                          calculate_half)

    constraints = model_zoo.KITE_CONSTRAINTS
    model = model_zoo.get_erhard_casadi_model(kite_parameter)
    f = model['f']
    stage_loss_function = ca.Function('stage_loss', [model['x'], model['u']], [-model['TF'] / t_final])

    # state and input constraints
    x_ub = [constraints['x_ub']] * (n + 1)
    x_lb = [constraints['x_lb']] * (n + 1)

    if use_stronger_constraints:
        x_ub = [50 / 180 * np.pi]
        x_ub.extend(constraints['x_ub'][1:])
        x_ub = [x_ub] * (n + 1)

    u_ub = [constraints['u_max']] * n
    u_lb = [constraints['u_min']] * n

    # height constraints
    h_lb = [[constraints['h_min']]] * n
    h_ub = [[np.inf]] * n

    # construct optimisation variables of the NLP using multiple shooting
    nx = len(constraints['x_ub'])
    nu = len(constraints['u_max'])
    x_sym, u_sym = [], []
    height = []

    # dynamic constraints
    dynamic_constraints = []
    dynamic_constraints_ub = [[0] * nx] * n
    dynamic_constraints_lb = [[0] * nx] * n

    # collect state loss
    state_loss = []

    x_sym.append(ca.MX.sym('X0', nx))

    for i in range(n):
        u_sym.append(ca.MX.sym(f'U{i}', nu))

        x = x_sym[-1]
        u = u_sym[-1]
        t = i * h

        height.append(model['height_function'](x_sym[-1]))

        current_loss = 0
        x_new = x
        for _ in range(number_of_steps):
            current_loss += stage_loss_function(x_new, u) * h / number_of_steps
            x_new = utils.rk4_step(f, t, x_new, u, h / number_of_steps)

        if residual_state_model is not None:
            x_new += residual_state_model.predict(x, u)
        if residual_thrust_model is not None:
            current_loss -= residual_thrust_model.predict(x, u) / t_final

        x_sym.append(ca.MX.sym(f'X{i + 1}', nx))
        state_loss.append(current_loss)

        dynamic_constraints.append(x_new - x_sym[-1])

    if calculate_half:
        periodic_constraint = [
            x_sym[0][0] - x_sym[-1][0],
            x_sym[0][1] + x_sym[-1][1],
            x_sym[0][2] + x_sym[-1][2]
        ]
    else:
        periodic_constraint = [x_sym[0] - x_sym[-1]]
    periodic_constraint_ub = [[0] * nx]
    periodic_constraint_lb = [[0] * nx]

    # delta u constraints
    if limit_delta_u:
        delta_u_constraints = [u_sym[i] - u_sym[i + 1] for i in range(len(u_sym) - 1)]
        delta_u_constraints_ub = [[2] * nu] * (n - 1)
        delta_u_constraints_lb = [[-2] * nu] * (n - 1)
    else:
        delta_u_constraints = []
        delta_u_constraints_ub = []
        delta_u_constraints_lb = []

    # put everything into NLP
    initial_values = np.concatenate((initial_state_trajectory + initial_input_trajectory))
    symbolic_variables = ca.vertcat(*(x_sym + u_sym))
    symbolic_variables_lb = np.concatenate((x_lb + u_lb))
    symbolic_variables_ub = np.concatenate((x_ub + u_ub))

    symbolic_constrains = ca.vertcat(*(dynamic_constraints + height + periodic_constraint + delta_u_constraints))
    constraints_lb = np.concatenate((dynamic_constraints_lb + h_lb + periodic_constraint_lb + delta_u_constraints_lb))
    constraints_ub = np.concatenate((dynamic_constraints_ub + h_ub + periodic_constraint_ub + delta_u_constraints_ub))

    # put t_final into NLP default initial and values and constraints are used
    if not fix_ts:
        symbolic_variables = ca.vertcat(symbolic_variables, t_final)
        initial_values = np.concatenate((initial_values, [t_final_init]))
        symbolic_variables_lb = np.concatenate((symbolic_variables_lb, [5]))
        symbolic_variables_ub = np.concatenate((symbolic_variables_ub, [60]))

    total_loss = 0
    for loss in state_loss:
        total_loss += loss

    # state problem and solve
    problem = {'f': total_loss,
               'x': symbolic_variables,
               'g': symbolic_constrains}
    solver = ca.nlpsol('solver', 'ipopt', problem)
    t_start = time.time()
    result = solver(x0=initial_values, lbx=symbolic_variables_lb, ubx=symbolic_variables_ub, lbg=constraints_lb,
                    ubg=constraints_ub)
    t_end = time.time()

    # fetch and return data
    get_trajectories = ca.Function('get_trajectories', [symbolic_variables],
                                   [ca.horzcat(*x_sym).T, ca.horzcat(*u_sym).T])
    x_opt, u_opt = get_trajectories(result['x'])
    x_opt, u_opt = x_opt.full(), u_opt.full()
    get_integrated_thrust_per_time_step = ca.Function('get_thrust_values', [symbolic_variables],
                                                      [-ca.vertcat(*state_loss)])
    integrated_thrust_per_time_step = get_integrated_thrust_per_time_step(result['x']).full().flatten()
    if not fix_ts:
        get_t_final_value = ca.Function('get_t_final_value', [symbolic_variables], [t_final])
        t_final_value = get_t_final_value(result['x']).full().item()
    else:
        t_final_value = t_final

    # generate symmetric second half
    if calculate_half:
        x_opt2 = x_opt * np.array([1, -1, -1])
        x_opt = np.vstack((x_opt[:-1], x_opt2))

        u_opt2 = u_opt * -1
        u_opt = np.vstack((u_opt, u_opt2))

        t_final_value *= 2

        integrated_thrust_per_time_step = np.concatenate(
            (integrated_thrust_per_time_step, integrated_thrust_per_time_step)) / 2

    doc = {'n': n,
           'kite_parameter': kite_parameter,
           'fix_ts': fix_ts,
           'date': datetime.today().strftime('%d-%m-%Y'),
           'solver time': t_end - t_start,
           'comment': comment,
           'calculate_half': calculate_half}

    return x_opt, u_opt, t_final_value, integrated_thrust_per_time_step, doc


def calculate_and_save_optimal_trajectory(name, n, kite_parameter, fix_ts=True, comment="", calculate_half=False,
                                          use_stronger_constraints=False,
                                          residual_state_model_name='',
                                          residual_thrust_model_name='', visualize=True, sub_folder_name='',
                                          use_initial_guess=True, number_of_steps=1, limit_delta_u=False):
    if residual_state_model_name:
        residual_state_model = model_zoo.load_residual_model(residual_state_model_name)
    else:
        residual_state_model = None
    if residual_thrust_model_name:
        residual_thrust_model = model_zoo.load_residual_model(residual_thrust_model_name + '_thrust')
    else:
        residual_thrust_model = None

    x, u, t_final, thrust, doc = _calculate_optimal_trajectory(n, kite_parameter,
                                                               fix_ts=fix_ts, comment=comment,
                                                               calculate_half=calculate_half,
                                                               use_stronger_constraints=use_stronger_constraints,
                                                               residual_state_model=residual_state_model,
                                                               residual_thrust_model=residual_thrust_model,
                                                               use_initial_guess=use_initial_guess,
                                                               number_of_steps=number_of_steps,
                                                               limit_delta_u=limit_delta_u)

    trajectory = utils.OptimalTrajectory(name, x, u, t_final, thrust, doc)
    trajectory.save(sub_folder_name=sub_folder_name)
    fig, _ = trajectory_plotting.plot_all_states_and_u(trajectory.x, t=trajectory.t, u=trajectory.u,
                                                       thrust=trajectory.integrated_thrust_per_time_step)
    if visualize:
        plt.show()
        print(t_final)
    plt.close(fig)
    return trajectory


if __name__ == '__main__':
    calculate_and_save_optimal_trajectory(name=_TRAJECTORY_ID, n=_N0, kite_parameter=_KITE_PARAMETER, fix_ts=_FIX_TS,
                                          comment=_COMMENT,
                                          calculate_half=_CALCULATE_HALF,
                                          residual_state_model_name=_RESIDUAL_MODEL_NAME,
                                          residual_thrust_model_name=_RESIDUAL_THRUST_MODEL_NAME,
                                          use_stronger_constraints=_USE_STRONGER_CONSTRAINTS,
                                          use_initial_guess=_USE_INIT,
                                          limit_delta_u=_LIMIT_DELTA_U)
