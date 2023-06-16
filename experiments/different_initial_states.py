""" Runs the simulation for different initial states.

"""

import simulate
from matplotlib import pyplot as plt
import utils
import path_handling
import pandas as pd
import numpy as np
from tqdm import trange


def _run_simulation_for_different_initial_values(controller_type, name_appendix='', horizon=5):
    simulation_parameter = simulate.get_simulation_parameter(120)
    folder = f'initial_values_{controller_type}' + name_appendix
    control_parameter = simulate.get_control_parameter(controller_type=controller_type,
                                                       prediction_model_name='physical',
                                                       reference_name='physical_trajectory')
    control_parameter['prediction_horizon'] = horizon

    theta_table = []
    phi_table = []
    psi_table = []
    names_table = []
    print('Run simulations with different initial states')
    pbar = trange(500, unit='Simulations')
    for i in pbar:
        utils.blockPrint()
        initial_value = utils.get_random_initial_value(400, 30)
        name = f'initial_value_{i}'
        names_table.append(name)
        simulation_parameter['initial_value'] = initial_value
        result = simulate.run_simulation(simulation_parameter, control_parameter, name=name,
                                         save_folder=folder,
                                         visualise=False,
                                         overwrite=False, dont_ask_user=True, notes='')
        # get correct initial value, if simulation already existed
        initial_value = result['x'][0]
        theta_table.append(initial_value[0])
        phi_table.append(initial_value[1])
        psi_table.append(initial_value[2])
        utils.enablePrint()
    pbar.close()

    extra_columns = {'theta_0': theta_table, 'phi_0': phi_table, 'psi_0': psi_table, 'name': names_table}
    simulate.make_table_of_simulations(f'initial_value_table_{controller_type}' + name_appendix, [folder],
                                       extra_colums=extra_columns)


def _create_success_plot(controller_type, name_appendix=''):
    table = pd.read_csv(path_handling.SIMULATION_DIR / f'initial_value_table_{controller_type}{name_appendix}.csv')
    success_initial_values = []
    failed_initial_values = []
    failed_names = []
    for name, theta, phi, psi, thrust, success_rate in table[
        ['name', 'theta_0', 'phi_0', 'psi_0', 'average_thrust', 'success_rate']].to_numpy():
        if thrust < 200000 or success_rate < 100:
            print(thrust)
            print(success_rate)
            failed_names.append(name)
            failed_initial_values.append([theta, phi, psi])
        else:
            success_initial_values.append([theta, phi, psi])
    success_initial_values = np.array(success_initial_values) * 180 / np.pi
    failed_initial_values = np.array(failed_initial_values) * 180 / np.pi
    fig, ax = utils.get_ax()
    ax.grid(True)
    ax.set_xlabel(r'$\phi$ in $[\deg]$')
    ax.set_ylabel(r'$\theta$ in $[\deg]$')
    cm = plt.cm.get_cmap('RdYlBu')
    h_min = 100
    L = 400
    phi_min = -75.6
    phi_max = 75.6
    phi = np.linspace(phi_min, phi_max, 1000)
    theta = np.arcsin(h_min / (np.cos(phi / 180 * np.pi) * L)) * 180 / np.pi
    ax.plot(phi, theta, 'k--')

    a = ax.scatter(success_initial_values[:, 1], success_initial_values[:, 0], c=success_initial_values[:, 2],
                   vmin=-180, vmax=180, s=10, cmap=cm)
    ax.scatter(failed_initial_values[:, 1], failed_initial_values[:, 0], c=failed_initial_values[:, 2],
               vmin=-180, vmax=180, s=100, cmap=cm, marker='+')
    for trajectory_name in ['physical_trajectory', 'plant_trajectory']:
        angles = utils.load_optimal_trajectories(trajectory_name).x * 180 / np.pi
        ax.scatter(angles[:, 1], angles[:, 0], c=angles[:, 2],
                   vmin=-180, vmax=180, s=20, cmap=cm, marker='*')
    cbar = plt.colorbar(a)
    cbar.ax.set_ylabel(r'$\psi$ in $[\deg]$', rotation=270)
    plt.tight_layout()
    fig.savefig(path_handling.SIMULATION_DIR / f'different_initial_values_{controller_type}{name_appendix}.png')
    print(
        'Print name of failed initial values.'
        'You might want to have a closer look at these simulations using view_simulation.py')
    print(failed_names)
    print(failed_initial_values)
    print(f'success percentage: {100*len(success_initial_values)/(len(success_initial_values)+len(failed_initial_values))}')
    return failed_initial_values / 180 * np.pi, failed_names


def _rerun_simulation_with_longer_prediction_horizon(initial_values, names, controller_type):
    simulation_parameter = simulate.get_simulation_parameter(120)
    folder = 'initial_values_rerun'
    control_parameter_1 = simulate.get_control_parameter(controller_type=controller_type,
                                                         prediction_model_name='physical',
                                                         reference_name='physical_trajectory')
    control_parameter_2 = simulate.get_control_parameter(controller_type=controller_type,
                                                         prediction_model_name='physical',
                                                         reference_name='physical_trajectory')
    control_parameter_2['prediction_horizon'] = 30

    for initial_value, name in zip(initial_values, names):
        simulation_parameter['initial_value'] = list(initial_value)
        simulate.run_simulation(simulation_parameter, control_parameter_1, name=name,
                                save_folder=folder,
                                visualise=False,
                                overwrite=False, dont_ask_user=True, notes='')
        simulate.run_simulation(simulation_parameter, control_parameter_2, name=name + '_long_horizon',
                                save_folder=folder,
                                visualise=False,
                                overwrite=False, dont_ask_user=True, notes='')


def main():
    controller_type = 'pf'
    _run_simulation_for_different_initial_values(controller_type, name_appendix='', horizon=5)
    bad_initial_values, bad_initial_values_names = _create_success_plot(controller_type, name_appendix='')
    # _rerun_simulation_with_longer_prediction_horizon(bad_initial_values, bad_initial_values_names, controller_type)


if __name__ == '__main__':
    main()
    plt.show()
