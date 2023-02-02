""" Runs or visualises the closed loop simulations.

This file defines the main functions of this project i.e. simulating the closed loop.
For this is loads and configures the controller and the simulator.
Then it saves the results and optionally visualises them.
Additionally, it defines default values for the simulator and the controller.

There are three ways to use this function.
1) Modify the main and run one specific simulation
2) use run_simulation() from another skript to perform the desired simulations
3) use visualize_result() in order to inspect an already existing simulation.

This file can be imported as a module and contains the following
functions:
    * run_simulation() - runs the closed loop simulation
    * visualize_result() - visualises an existing simulation
    * make_table_of_simulations() - creates a table of all simulation results in a given folder
    * get_simulation_parameter() - returns the default simulation parameter set
    * get_control_parameter() - returns the default control parameter set

"""

from controller import controller_zoo
from simulation import towing_kite_simulator
from matplotlib import pyplot as plt
from visualisation import trajectory_plotting
import utils
import path_handling
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from interfaces import SimulationResult
from interfaces import SimulationParameter
from interfaces import ControlParameter
import os
import model_zoo
from scipy.signal import find_peaks

import pandas as pd
from typing import List, Dict, Tuple


def _get_wind_and_drift_function(sim_parameter):
    magnitude = sim_parameter['magnitude']
    frequency = sim_parameter['wind_frequency']
    delay = sim_parameter['delay']
    tau = sim_parameter['tau']
    index_wind_function = sim_parameter['wind_function_index']
    index_drift_function = sim_parameter['drift_function_index']

    DRIFT_FUNCTIONS = [None, lambda t: min(1 + 0.001 * t, 2.5)]  # drift for E0
    WIND_FUNCTIONS = [None, lambda t: np.sin(t * 2 * np.pi * frequency) * magnitude / 15 + 1,
                      # todo get normalisation factor from parameter
                      lambda t: 1 + magnitude / 15 * 1 / (1 + np.exp(-(t - delay) / tau))]
    return WIND_FUNCTIONS[index_wind_function], DRIFT_FUNCTIONS[index_drift_function]


def _ask_user_overwrite():
    user_input = input(
        'simulation id already exits!\nenter "y" if you want to override it.'
        '\nenter anything else to show the results of the existing simulation.\n')

    if user_input != 'y':
        overwrite = False
    else:
        overwrite = True

    return overwrite


def _find_index_of_first_and_last_period(result):
    k_last_period = result['k_index'][-1]
    difference = np.linalg.norm(result['x'] - result['x'][k_last_period], axis=1)
    peaks = find_peaks(-difference)[0]
    k_first_period = k_last_period
    for i in range(len(peaks)):
        k_first_period = peaks[i]
        if difference[k_first_period] < 1e-1:
            break
    if k_first_period == k_last_period:
        print('k_last_period == k_last_period!!!!! they are set to default value.')
        k_first_period = 0
        k_last_period = len(result['x']) - 1
    n_peaks = len(peaks)
    return k_first_period, k_last_period, n_peaks


def _calculate_average_thrust_and_window(result):
    k_first_period, k_last_period, n_peaks = _find_index_of_first_and_last_period(result)

    thrust = result['moment_values'][k_first_period:k_last_period]
    t = result['t'][k_first_period:k_last_period]
    t_first_period = t[0]
    t_last_period = t[-1]
    delta_t = t[1:] - t[:-1]
    mid_value = (thrust[:-1] + thrust[1:]) / 2
    average_thrust = (delta_t * mid_value).sum() / (t_last_period - t_first_period)
    return average_thrust, t_first_period, t_last_period, n_peaks


def _save_simulation(result: SimulationResult, control_parameter, sim_parameter, path, notes, simulation_duration,
                     name, metrics):
    simulation_date = datetime.today().strftime('%d-%m-%Y')

    log = f'This simulation was performed on the {simulation_date} and needed {simulation_duration:.3f}s ' \
          f'to be performed.\nThe average / maximum computation time per time step was: ' \
          f'{result["computation_times"].mean():.3f}s / {np.max(result["computation_times"]):.3f}s\n' \
          f'\nAdditional Notes:\n{notes}'
    if not path.is_dir():
        os.makedirs(path)
    # save simulation
    with open(path / 'log.txt', 'w') as f:
        f.write(log)
    with open(path / 'result.pkl', 'wb') as output:
        pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)
    with open(path / 'controller_parameter.json', 'w') as output:
        json.dump(control_parameter, output)
    with open(path / 'simulation_parameter.json', 'w') as output:
        json.dump(sim_parameter, output)
    with open(path / 'metrics.json', 'w') as output:
        json.dump(metrics, output)

    # store some results in easier to read files
    trajectory_df = pd.DataFrame({
        't': result['t'][result['k_index']],
        'integrated_thrust_per_time_step': np.append(result['integrate_time_step_value'], np.nan),
        'u': np.append(result['u'][result['k_index'][:-1]], np.nan),  # append value to match length
        'theta_measure': result['measurements'][:, 0],
        'phi_measure': result['measurements'][:, 1],
        'psi_measure': result['measurements'][:, 2],
        'theta': result['x'][result['k_index']][:, 0],
        'phi': result['x'][result['k_index']][:, 1],
        'psi': result['x'][result['k_index']][:, 2],
    })
    solution_state_trajectories = utils.get_solution_state_trajectories_from_result(result)
    for state_number, state in enumerate(['theta', 'phi', 'psi']):
        for prediction_step in range(1, solution_state_trajectories.shape[1]):  # 0-step pred is equal to measurement
            trajectory_df[f'{state}_{prediction_step}_step_prediction'] = np.append(
                solution_state_trajectories[:, prediction_step, state_number], np.nan)
    trajectory_df.to_csv(path / 'result.csv', index=False)

    # save png from closed loop simulation
    reference_trajectory = utils.load_optimal_trajectories(control_parameter['reference'])
    plant_trajectory = utils.load_optimal_trajectories('plant_trajectory')
    physical_trajectory = utils.load_optimal_trajectories('physical_trajectory')
    fig, ax = trajectory_plotting.plot_phi_theta_plane(
        [result['x'], plant_trajectory.x, physical_trajectory.x, reference_trajectory.x],
        labels=['closed loop', 'optimal plant', 'optimal physical', 'reference'])
    fig.savefig(path.parent / f'{name}.png')
    plt.close(fig)

    return result


def _calculate_metrics(result, control_parameter):
    # find window of full periods after transient phase and calculate thrust
    average_thrust, t_first_period, t_last_period, n_peaks = _calculate_average_thrust_and_window(result)

    # look at heights
    # todo get height from parameters
    height = utils.transform_to_cartesian(result['measurements'], 400)[:, 2]
    true_height = utils.transform_to_cartesian(result['x'], 400)[:, 2]
    min_height_measured = height.min()
    min_height = true_height.min()

    model_plant_missmatch = [np.linalg.norm(result['measurements'][i + 1] - result['info'][i]['x_sol'][1]) for i in
                             range(len(result['measurements']) - 1)]
    success = [result['info'][i]['success'] for i in range(len(result['measurements']) - 1)]

    controller_type = control_parameter['type']
    # get tracking error and period duration
    reference_trajectory = utils.load_optimal_trajectories(control_parameter['reference'])

    if controller_type == 'pf':
        virtual_states = np.array([result['info'][i]['z_sol'][0, 3:] for i in range(len(result['info']))])
        average_period_duration = result["t"][-1] / virtual_states[-1, 0]
        reference_path = utils.ControlPath(reference_trajectory)
        reference_states = reference_path(virtual_states[:, 0])
    else:
        average_period_duration = reference_trajectory.t[-1]
        n = len(result['measurements'])
        reference_states = np.resize(reference_trajectory.x[:-1], (n - 1, reference_trajectory.x.shape[1]))

    dx_measure = reference_states - result['measurements'][:-1]
    dx_true = reference_states - result['x'][result['k_index']][:-1]

    dx_cart_measure = utils.transform_to_cartesian(reference_states, 400) - utils.transform_to_cartesian(
        result['measurements'][:-1], 400)
    dx_cart = utils.transform_to_cartesian(reference_states, 400) - utils.transform_to_cartesian(
        result['x'][result['k_index']][:-1], 400)

    metrics = {
        'average_thrust': average_thrust,
        'average_thrust_not_corrected': result['integrate_value'] / result['t'][-1],
        'min_height_measured': min_height_measured,
        'min_height': min_height,
        'average_model_plant_missmatch': np.mean(model_plant_missmatch),
        'success_rate': np.mean(success) * 100,
        't_first_period': t_first_period,
        't_last_period': t_last_period,
        'average_period_duration_reference': average_period_duration,
        'average_tacking_error_measured_in_deg': np.linalg.norm(dx_measure, axis=1).mean() * 180 / np.pi,
        'average_tacking_error_in_deg': np.linalg.norm(dx_true, axis=1).mean() * 180 / np.pi,
        'average_tacking_error_measured_in_m': np.linalg.norm(dx_cart_measure, axis=1).mean(),
        'average_tacking_error_in_m': np.linalg.norm(dx_cart, axis=1).mean(),
    }
    return metrics


def make_table_of_simulations(save_name: str, folder_names: List[str],
                              extra_colums: Dict[str, List] = None) -> pd.DataFrame:
    """ Creates a table of the main simulation results within the given folders

    Use this to get an overview of the results if multiple simulations should be analysed.

    Parameters
    ----------
    save_name : str
        table will be saved as $(SIMULATION_DIR)/save_name.csv
    folder_names : List[str]
        all simulations stored in the folders in that list will be used. (relative to $(SIMULATION_DIR))
    extra_colums : Dict[str, List]
        extra colums that should be added to the table. It must contain the keyword "name" which contains a list of
        all simulation names in the folders. These names are used to match the order of the values stored with all other
        keywords to the collected simulations. Then all keyword and their values are added to the table.

    Returns
    -------
    pd.DataFrame
        table with the collected simulation results.
    """
    metric_table = pd.DataFrame()
    for folder_name in folder_names:
        path = path_handling.SIMULATION_DIR / folder_name
        for folder in path.iterdir():
            if folder.is_dir():
                result, sim_param, control_param, metrics = load_simulation(folder)
                save_dict = {'folder': folder_name, 'name': folder.name, 'type': control_param['type'],
                             'reference': control_param['reference'],
                             'prediction_model': control_param['prediction_model']}
                save_dict.update(metrics)
                metric_table = pd.concat([metric_table, pd.DataFrame([save_dict])], ignore_index=True)
    if extra_colums is not None:
        names = extra_colums['name']
        new_order = [names.index(name) for name in metric_table['name']]
        for key, item in extra_colums.items():
            if key == 'name':
                continue
            metric_table[key] = [item[i] for i in new_order]
    metric_table.to_csv(path_handling.SIMULATION_DIR / f'{save_name}.csv', index=False)
    return metric_table


def load_simulation(path):
    """ loads a simulation result

    Parameters
    ----------
    path : Path
        path where the simulation is stored

    Returns
    -------
    Tuple[SimulationResult, ControlParameter, SimulationParameter, Dict]
        loads the simulation result, control-/ and simulation-parameter and some computed metrics
    """
    with open(path / 'result.pkl', 'rb') as inp:
        result: SimulationResult = pickle.load(inp)
    with open(path / 'controller_parameter.json', 'rb') as input_file:
        control_parameter = json.load(input_file)
    with open(path / 'simulation_parameter.json', 'rb') as input_file:
        simulation_parameter = json.load(input_file)
    with open(path / 'metrics.json', 'rb') as input_file:
        metrics = json.load(input_file)
    return result, simulation_parameter, control_parameter, metrics


def run_simulation(sim_parameter: SimulationParameter, control_parameter: ControlParameter, name: str, save_folder: str,
                   notes: str, visualise: bool, overwrite: bool = False,
                   dont_ask_user: bool = False) -> SimulationResult:
    """Runs and saves a closed-loop simulation.

    This function will run and save a closed loop simulation. The results are stored in $(SIMULATION_FOLDER)

    Parameters
    ----------
    sim_parameter : SimulationParameter
        parameter settings of the simulator
    control_parameter : ControlParameter
        parameter settings of the controller
    name : str
        name or id of the simulation (used as its saving name)
    save_folder : str
        folder where the simulation should be saved (relative to $(SIMULATION_FOLDER))
    notes : str
        notes that will also be stored and that can help the user to document the simulations better
    visualise : bool
        whether the simulation result should be plotted
    overwrite : bool (default = False)
        whether an existing simulation with the name id should be overwritten by default. If false the user is asked.
    dont_ask_user : bool (default = False)
        if overwrite is True usually the user is ask whether they want to overwrite the simulation.
        if this flag is True the user is not asked and the simulation will not be overwritten.

    """
    wind_function, drift_function = _get_wind_and_drift_function(sim_parameter)

    path = path_handling.SIMULATION_DIR / save_folder / name
    if path.is_dir() and not overwrite:
        if dont_ask_user:
            calculate = False
            print(f'simulation {name} in folder {save_folder} already exists. Simulation aborted.')
        else:
            calculate = _ask_user_overwrite()
    else:
        calculate = True

    controller = controller_zoo.get_controller(control_parameter, wind_function=wind_function)
    simulator = towing_kite_simulator.get_kite_simulator(controller,
                                                         noise_std_deg=sim_parameter['noise_std_deg'],
                                                         wind_function=wind_function,
                                                         plant_number=sim_parameter['plant_number'],
                                                         drift_function=drift_function)
    if sim_parameter['initial_value'] is None:
        x0 = controller.get_x_init()  # this is the initial value expected by the controller (e.g. start of reference)
    else:
        x0 = np.array(sim_parameter['initial_value'])
        controller.set_x_init(x0)

    if calculate:
        start_time = time.time()
        result = simulator.run_simulation(sim_parameter['simulation_time'], x0, verbose=1,
                                          delay_input=sim_parameter['delay_u'])
        simulation_duration = time.time() - start_time

        metrics = _calculate_metrics(result, control_parameter)
        _save_simulation(result, control_parameter, sim_parameter, path, notes, simulation_duration, name, metrics)

        if visualise:
            visualize_result(name, save_folder)
        else:
            print(f'Average Thrust: {metrics["average_thrust"]}')
        return result
    else:
        print('simulation aborted. Use result of existing simulation. Settings might be different!!')
        return load_simulation(path_handling.SIMULATION_DIR / save_folder / name)[0]


def visualize_result(name: str, save_folder: str):
    """visualises a simulation result.

    plots some basic information of the simulation.

    Parameters
    ----------
    name : str
        name or id of the simulation which should be loaded
    save_folder : str
        folder where the simulation is located (relative to $(SIMULATION_DIR))

    """
    path = path_handling.SIMULATION_DIR / save_folder / name
    result, sim_parameter, control_parameter, metrics = load_simulation(path)

    # load trajectories as reference for plots
    plant_trajectory = utils.load_optimal_trajectories('plant_trajectory')
    physical_trajectory = utils.load_optimal_trajectories('physical_trajectory')
    reference_trajectory = utils.load_optimal_trajectories(control_parameter['reference'])

    # plot phase diagram
    fig, _ = trajectory_plotting.plot_phi_theta_plane(
        [result['x'], plant_trajectory.x, physical_trajectory.x, reference_trajectory.x],
        labels=['closed loop', 'optimal plant', 'optimal physical', 'reference'])
    fig.canvas.manager.set_window_title('phase_diagram')

    # plot measured and actual height
    height = utils.transform_to_cartesian(result['measurements'], 400)[:, 2]
    true_height = utils.transform_to_cartesian(result['x'], 400)[:, 2]
    print(f'min height measured {height.min()}')
    print(f'min height {true_height.min()}')
    print(f'Average Thrust: {metrics["average_thrust"]}')

    plt.figure('height')
    plt.plot(np.linspace(0, result['t'][-1], len(height)), height)
    plt.plot(result['t'], true_height)
    plt.title('height')
    plt.grid()

    # plot loss
    loss = np.array([result['info'][i]['loss'] for i in range(1, len(result['info']))])
    plt.figure('loss')
    plt.plot(loss)
    plt.xlabel('t')
    plt.title('loss')
    plt.grid()

    # plot 3d path
    fig = trajectory_plotting.plot_x_y_z_trajectory([result['x'], reference_trajectory.x], 400,
                                                    labels=['closed-loop', 'reference'])
    fig.canvas.manager.set_window_title('cartesian plot')

    # plot comparison of reference trajectory to closed loop
    n = len(result['measurements'])
    x_ref = np.resize(reference_trajectory.x[:-1], (n, reference_trajectory.x.shape[1]))
    u_ref = np.resize(reference_trajectory.u, (n - 1, reference_trajectory.u.shape[1]))
    thrust_ref = np.resize(reference_trajectory.integrated_thrust_per_time_step,
                           (n - 1, reference_trajectory.u.shape[1]))
    t_ref = np.arange(n) * model_zoo.ERHARD_Ts
    fig, axs = trajectory_plotting.plot_all_states_and_u([result['x'], x_ref, result['measurements']],
                                                         t=[result['t'], t_ref, t_ref],
                                                         u=[result['u'], u_ref, result['u_controller']],
                                                         thrust=[result['moment_values'] * 0.27, thrust_ref,
                                                                 result['integrate_time_step_value']],
                                                         labels=['simulation', 'offline trajectory', 'measurements'])
    fig.canvas.manager.set_window_title('closed_loop and offline trajectory')
    # plot window considered for average thrust
    average_thrust, t_first_period, t_last_period, n_peaks = _calculate_average_thrust_and_window(result)
    for ax in axs:
        ax.axvline(x=t_first_period)
        ax.axvline(x=t_last_period)

    # plot model plant missmatch
    model_plant_missmatch = [np.linalg.norm(result['measurements'][i + 1] - result['info'][i]['x_sol'][1]) for i in
                             range(len(result['measurements']) - 1)]

    model_plant_missmatch_final = [np.linalg.norm(result['measurements'][i + 5] - result['info'][i]['x_sol'][5]) for i
                                   in
                                   range(len(result['measurements']) - 5)]

    plt.figure('model-plant mismatch')
    plt.plot(model_plant_missmatch, label='one_step')
    plt.plot(model_plant_missmatch_final, label='five_step')
    plt.legend()
    plt.grid()
    plt.title('model-plant mismatch')

    # plot success over time
    success = [result['info'][i]['success'] for i in range(len(result['measurements']) - 1)]
    plt.figure('feasible')
    plt.plot(success)
    plt.title('feasible')

    # only relevant for pf controller
    if control_parameter['type'] == 'pf':
        # get virtual states
        virtual_states = np.array([result['info'][i]['z_sol'][0, 3:] for i in range(len(result['info']))])

        # plot virtual states
        time_vector = result['t'][result['k_index'][:-1]]
        plt.figure('virtual state 1')
        plt.plot(time_vector, virtual_states[:, 0])
        plt.xlabel('t')
        plt.title('virtual state 1')
        print(
            f'final z1 = {virtual_states[-1, 0]}, average period duration = {result["t"][-1] / virtual_states[-1, 0]}')

        plt.figure('virtual state 2')
        plt.plot(time_vector, virtual_states[:, 1])
        plt.xlabel('t')
        plt.title('virtual state 2')

        plt.figure('virtual state 3')
        plt.plot(time_vector, virtual_states[:, 2])
        plt.xlabel('t')
        plt.title('virtual state 3')

        # plot reference compared to true states
        reference_path = utils.ControlPath(reference_trajectory)
        reference_states = reference_path(virtual_states[:, 0])
        dx = reference_states - result['measurements'][:-1]
        dx_cart = utils.transform_to_cartesian(reference_states, 400) - utils.transform_to_cartesian(
            result['x'][result['k_index']][:-1], 400)

        plt.figure('tracking error')
        plt.plot(virtual_states[:, 0], dx)
        plt.title('tracking error')
        plt.xlabel('z1')

        dx_2 = reference_states - result['x'][result['k_index']][:-1]
        print(f'average tacking error in deg : {np.linalg.norm(dx, axis=1).mean() * 180 / np.pi}')
        print(f'average tacking error in deg (sim) : {np.linalg.norm(dx_2, axis=1).mean() * 180 / np.pi}')
        print(f'average tacking error in m : {np.linalg.norm(dx_cart, axis=1).mean()}')

        plt.figure('reference')
        plt.plot(virtual_states[:, 0], result['measurements'][:-1], label='m')
        plt.plot(virtual_states[:, 0], reference_states, label='ref')
        plt.xlabel('z1')
        plt.legend()
        plt.grid()
        plt.title('reference')

        # plot computation time over s
        plt.figure('computation_time')
        plt.plot(virtual_states[:, 0], result['computation_times'])
        plt.xlabel('z1')
        plt.title('computation_time')

    plt.show()


def get_simulation_parameter(simulation_time: float = 120) -> SimulationParameter:
    """ returns the default simulation parameter

    This returns the default simulation parameter. The user can adjust the simulation time. All other changes such
    as the noise level the initial value or the wind function and so on the simulation parameter must be changed by hand

    Parameters
    ----------
    simulation_time : float (default = 120)
        length of the simulation in seconds

    Returns
    -------
    SimulationParameter
        default simulation parameter
    """
    DEFAULT_SIMULATION_PARAMETER = {
        'initial_value': None,  # if None the initial state of the reference of the controller is used.
        'simulation_time': simulation_time,  # total simulation time
        'noise_std_deg': list(model_zoo.ERHARD_NOISE_STD_DEG_MEDIUM),  # std dev in deg. Set None to deactivate noise.
        'delay_u': False,  # if True means that only measurements up to k-1 are available to the controller
        'wind_function_index': 0,  # index of the wind function
        'wind_frequency': 0.1,  # frequency for first wind function
        'magnitude': 0.2,  # magnitude of the wind function (relative to constant wind)
        'delay': 60,  # delay time for second wind function
        'tau': 0.5,  # steepness for second wind function
        'drift_function_index': 0,  # index of the drift function
        'plant_number': 1,  # index of the plant that is used
    }

    return DEFAULT_SIMULATION_PARAMETER


def get_control_parameter(controller_type: str = 'pf', prediction_model_name: str = 'physical',
                          reference_name: str = 'physical_trajectory',
                          activate_online_learning: bool = False) -> ControlParameter:
    """ returns the default control parameter

    This returns the default control parameter. The user can adjust some basic settings. All other changes such
    as the control parameter, online learning settings, residual model name and so on must be changed by hand

    Parameters
    ----------
    controller_type : str (default = "pf")
        type of the controller (either "pf" for path following or "tt" for trajectory tracking)
    prediction_model_name : str (default = "physical")
        name of the prediction model (either "physical", "hybrid" or "plant")
    reference_name : str (default = "physical_trajectory")
        name of the reference (either "physical_trajectory", "plant_trajectory", "hybrid_trajectory",
        "hybrid_trajectory_half", or any self generated optimal trajectory.)
    activate_online_learning : bool (default = False)
        whether online learning should be applied.

    Returns
    -------
    ControlParameter
        default control parameter
    """
    if controller_type == 'tt':
        tt_parameter = {
            'hmin': 95,
            'Q': [[100, 0, 0],
                  [0, 100, 0],
                  [0, 0, 100]],
            'final_weight': 1,
            'R': 1,
            'S': 1
        }
    else:
        tt_parameter = None
    if controller_type == 'pf':
        pf_parameter = {
            'hmin': None,
            'Q': [[2000, 0, 0],
                  [0, 100, 0],
                  [0, 0, 100]],
            'final_weight': 1,
            'S': [[0.2, 0], [0, 0]],
            'n_virtual': 3,
            'virtual_state_lb': [0., 0, -10],
            'virtual_state_ub': [np.inf, 0.25, 10],
            'virtual_input_lb': -10,
            'virtual_input_ub': 10,
        }
    else:
        pf_parameter = None

    if activate_online_learning:
        online_learning_parameter = {
            'active_algorithm': 0,  # online learning method
            'monitor_algorithm': [],
            'n_max_data_points': 100,
            'keep_offline_data': True,
            'threshold_x_err': [0.0, 0.0, 0.0],
            'threshold_variance': [1e-3, 1e-3, 1e-3],
        }
    else:
        online_learning_parameter = None
    control_parameter: ControlParameter = {
        'type': controller_type,  # 'pf' or 'tt' or 'uf' (for uf only 'reference' is of interest)
        'use_soft_constraint': True,  # whether height constraint is soft
        'time_variant': False,  # wind function is known to the controller
        'wind_function': 0,  # wind function of the controller (only applied if time_variant is True)
        'prediction_horizon': 5,
        'prediction_model': prediction_model_name,  # 'physical', 'hybrid' or 'plant'
        'residual_model': 'default_mlp',  # name of the residual model
        'reference': reference_name,  # name of the trajectory that is used as reference
        'online_learning_settings': online_learning_parameter,
        'discretization_method': 'direct_multiple_shooting',  # 'direct_multiple_shooting' or 'direct_single_shooting'
        'integration_method': 'rk4',  # 'rk4' or 'euler'
        'integration_steps': 1,  # number of integration steps per time step
        'pf_parameter': pf_parameter,
        'tt_parameter': tt_parameter
    }
    return control_parameter


if __name__ == '__main__':
    _sim_parameter = get_simulation_parameter(simulation_time=100)
    _control_parameter = get_control_parameter(controller_type='pf', prediction_model_name='physical',
                                               reference_name='physical_trajectory')

    run_simulation(sim_parameter=_sim_parameter,
                   control_parameter=_control_parameter,
                   name='test',
                   save_folder='test',
                   notes='Hello', visualise=True)
