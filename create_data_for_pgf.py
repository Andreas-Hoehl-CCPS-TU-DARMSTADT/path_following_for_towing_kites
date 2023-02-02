""" This file generated the data needed to plot the results shown in the paper.

Note that it only loads the existing simulation results. So run the setup and the experiments first.
"""

import simulate
import pandas as pd
import path_handling
import numpy as np
import utils

# minimal height phase plot
h_min = 100
L = 400
phi_min = -75.5
phi_max = 75.5
phi = np.linspace(phi_min, phi_max, 100)
theta = np.arcsin(h_min / (np.cos(phi / 180 * np.pi) * L)) * 180 / np.pi

height_df = pd.DataFrame({
    'theta': theta,
    'phi': phi
})
with open(path_handling.PGF_DIR / f'h_min_100.txt', 'w') as f:
    dfAsString = height_df.to_string(header=True, index=False)
    f.write(dfAsString)

# initial value plot
initial_values_pf = pd.read_csv(path_handling.SIMULATION_DIR / 'initial_value_table_pf.csv')

success_initial_values = []
failed_initial_values = []
for name, theta, phi, psi, thrust, success_rate in initial_values_pf[
    ['name', 'theta_0', 'phi_0', 'psi_0', 'average_thrust', 'success_rate']].to_numpy():
    if thrust < 520000 or success_rate < 100:
        failed_initial_values.append([theta, phi, psi])
    else:
        success_initial_values.append([theta, phi, psi])
success_initial_values = np.array(success_initial_values) * 180 / np.pi
failed_initial_values = np.array(failed_initial_values) * 180 / np.pi

for values, name in zip([success_initial_values, failed_initial_values], ['good', 'bad']):
    initial_values_df = pd.DataFrame({
        'theta': values[:, 0],
        'phi': values[:, 1],
        'psi': values[:, 2]
    })

    with open(path_handling.PGF_DIR / f'{name}_initial_values.txt', 'w') as f:
        dfAsString = initial_values_df.to_string(header=True, index=False)
        f.write(dfAsString)

# 3d trajectory tracking plot
for N, name in zip([65, 70, 73, 88], ['very_fast', 'fast', 'good', 'original']):
    result, _, _, _ = simulate.load_simulation(path_handling.SIMULATION_DIR / 'resampled_trajectories' /
                                               f'physical_trajectory_resampled_to_{N}_physical_prediction_model')
    x_cart = utils.transform_to_cartesian(result['x'][result['k_index']], 400)
    tt_3d_plot_df = pd.DataFrame({
        'x': x_cart[:, 0],
        'y': x_cart[:, 1],
        'z': x_cart[:, 2]
    })

    with open(path_handling.PGF_DIR / f'{name}_tt_3d_plot_df.txt', 'w') as f:
        dfAsString = tt_3d_plot_df.to_string(header=True, index=False)
        f.write(dfAsString)

physical_trajectory = utils.load_optimal_trajectories('physical_trajectory')
x_cart = utils.transform_to_cartesian(physical_trajectory.x, 400)
tt_3d_plot_df = pd.DataFrame({
    'x': x_cart[:, 0],
    'y': x_cart[:, 1],
    'z': x_cart[:, 2]
})

with open(path_handling.PGF_DIR / f'reference_tt_3d_plot_df.txt', 'w') as f:
    dfAsString = tt_3d_plot_df.to_string(header=True, index=False)
    f.write(dfAsString)

# default path following reference
result, _, _, _ = simulate.load_simulation(path_handling.SIMULATION_DIR / 'default_parameter' /
                                           'physical_prediction_model_pf' / 'physical_trajectory')
reference_speed = np.array([result['info'][k]['z_sol'][0, 4] for k in range(len(result['info']))])
# average is about 0.0582
# for physical we get 1/24.04=0.0416
reference_speed_df = pd.DataFrame({
    't': np.arange(len(reference_speed)) * 0.27,
    'z2': reference_speed,
})

with open(path_handling.PGF_DIR / f'reference_speed.txt', 'w') as f:
    dfAsString = reference_speed_df.to_string(header=True, index=False)
    f.write(dfAsString)

# initial value phase plot
for controller_type, idx in zip(['tt', 'pf'], [25, 45]):
    result, _, _, _ = simulate.load_simulation(path_handling.SIMULATION_DIR / f'initial_values_{controller_type}' /
                                               f'initial_value_{idx}')
    states = result['x'][result['k_index']] * 180 / np.pi
    states_df = pd.DataFrame({
        'theta': states[:, 0],
        'phi': states[:, 1],
    })

    with open(path_handling.PGF_DIR / f'initial_value_phase_plot_{controller_type}.txt', 'w') as f:
        dfAsString = states_df.to_string(header=True, index=False)
        f.write(dfAsString)

    initial_states_df = pd.DataFrame({
        'theta': states[0:1, 0],
        'phi': states[0:1, 1],
        'psi': states[0:1, 2]
    })

    with open(path_handling.PGF_DIR / f'initial_state_initial_value_phase_plot_{controller_type}.txt', 'w') as f:
        dfAsString = initial_states_df.to_string(header=True, index=False)
        f.write(dfAsString)

# reference speed wind scenario
result, _, _, _ = simulate.load_simulation(path_handling.SIMULATION_DIR / 'harmonic_wind_pf' / 'freq_0.01_mag_10')
reference_speed = np.array([result['info'][k]['z_sol'][0, 4] for k in range(len(result['info']))])
reference_speed_df = pd.DataFrame({
    't': np.arange(len(reference_speed)) * 0.27,
    'z2': reference_speed,
})

with open(path_handling.PGF_DIR / f'reference_speed_harmonic_wind.txt', 'w') as f:
    dfAsString = reference_speed_df.to_string(header=True, index=False)
    f.write(dfAsString)
