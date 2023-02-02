""" This file should be used to load a controller with the desired settings.

"""
import model_zoo
import utils
from controller import my_MPC
from controller.basic_contoller import UForwardController, Controller
import numpy as np
from utils import ControlPath
from interfaces import ControlParameter
from typing import Callable


def get_controller(controller_options: ControlParameter, wind_function: Callable[[float], float]):
    """ loads the controller

    Parameters
    ----------
    controller_options : ControlParameter
        controller settings
    wind_function : Callable[[float], float]
        see model_zoo.get_erhard_dgl_and_thrust_function

    Returns
    -------
    Controller
        controller with the desired settings

    """
    time_variant = controller_options['time_variant']

    N = controller_options['prediction_horizon']
    trajectory = utils.load_optimal_trajectories(controller_options['reference'])

    integration_method = controller_options['integration_method']
    integration_steps = controller_options['integration_steps']

    online_learning_options = controller_options['online_learning_settings']

    discretization_method = controller_options['discretization_method']

    use_soft_constraint = controller_options['use_soft_constraint']

    n_virtual_states = controller_options['pf_parameter']['n_virtual'] if controller_options['type'] == 'pf' else None
    prediction_model = model_zoo.get_Erhard_prediction_model(system_name=controller_options['prediction_model'],
                                                             steps=integration_steps,
                                                             discretization_method=integration_method,
                                                             n_virtual_states=n_virtual_states,
                                                             time_variant=time_variant,
                                                             wind_function=wind_function,
                                                             state_GP_name=controller_options['residual_model'],
                                                             thrust_GP_name=None)

    if controller_options['type'] == 'tt':
        return _get_trajectory_tracking_controller(N,
                                                   discretization_method,
                                                   prediction_model, trajectory.x, trajectory.u,
                                                   controller_options['tt_parameter'], use_soft_constraint,
                                                   online_learning_options)

    elif controller_options['type'] == 'pf':
        path = ControlPath(trajectory)
        return _get_path_following_controller(N,
                                              discretization_method,
                                              prediction_model, path, controller_options['pf_parameter'],
                                              use_soft_constraint,
                                              online_learning_options)

    elif controller_options['type'] == 'uf':
        return UForwardController(trajectory.u, trajectory.x[0])


def _get_path_following_controller(N, discretization_method, extended_prediction_model,
                                   path, pf_options, use_soft_constraint, online_learning_options):
    final_weight = pf_options['final_weight']
    Q = np.array(pf_options['Q'])
    hmin = pf_options['hmin']

    constraints = model_zoo.KITE_CONSTRAINTS

    if hmin is None:
        height_function = None
        height_function_terminal = None
        height_bounds = None
    else:
        height_function = lambda x, u: extended_prediction_model.casadi_model['height_function'](x)
        height_function_terminal = lambda x: extended_prediction_model.casadi_model['height_function'](x)
        height_bounds = ([hmin], [np.inf])

    NLP_constraints: utils.NLPConstraints = dict(xlb=constraints['x_lb'] + pf_options['virtual_state_lb'],
                                                 xub=constraints['x_ub'] + pf_options['virtual_state_ub'],
                                                 xlb_terminal=constraints['x_lb'] + pf_options['virtual_state_lb'],
                                                 xub_terminal=constraints['x_ub'] + pf_options['virtual_state_ub'],
                                                 ulb=constraints['u_min'] + [pf_options['virtual_input_lb']],
                                                 uub=constraints['u_max'] + [pf_options['virtual_input_ub']],
                                                 stage_constrains=height_function,
                                                 stage_constrains_bounds=height_bounds,
                                                 terminal_constrains=height_function_terminal,
                                                 terminal_constrains_bounds=height_bounds)
    S = np.array(pf_options['S']) if pf_options['S'] is not None else None
    controller = my_MPC.PathFollowingMPC(Q, S, final_weight, use_soft_constraint, extended_prediction_model, N,
                                         NLP_constraints, path, discretization=discretization_method,
                                         online_learning_options=online_learning_options)

    return controller


def _get_trajectory_tracking_controller(N, discretization_method, prediction_model,
                                        x_opt, u_opt, tt_options, use_soft_constraint, online_learning_options):
    final_weight = tt_options['final_weight']
    Q = np.array(tt_options['Q'])
    h_min = tt_options['hmin']
    constrains = model_zoo.KITE_CONSTRAINTS

    height_function = lambda x, u: prediction_model.casadi_model['height_function'](x)
    height_function_terminal = lambda x: prediction_model.casadi_model['height_function'](x)
    height_bounds = ([h_min], [np.inf])

    NLP_constraints: utils.NLPConstraints = dict(xlb=constrains['x_lb'],
                                                 xub=constrains['x_ub'],
                                                 xlb_terminal=constrains['x_lb'],
                                                 xub_terminal=constrains['x_ub'],
                                                 ulb=constrains['u_min'],
                                                 uub=constrains['u_max'],
                                                 stage_constrains=height_function,
                                                 stage_constrains_bounds=height_bounds,
                                                 terminal_constrains=height_function_terminal,
                                                 terminal_constrains_bounds=height_bounds)

    R = np.array(tt_options['R'])
    S = np.array(tt_options['S']) if tt_options['S'] is not None else None
    controller = my_MPC.TrajectoryTrackingMPC(Q, S, R, final_weight, use_soft_constraint, prediction_model, N,
                                              NLP_constraints, x_opt, u_opt,
                                              discretization=discretization_method,
                                              online_learning_options=online_learning_options)

    return controller
