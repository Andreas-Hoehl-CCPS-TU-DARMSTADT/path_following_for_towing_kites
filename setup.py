"""This file generates everything needed to run your experiments or the given examples

Run this script to create the folder structure describes in the README.md.
Then it calculates the optimal physical and plant trajectories.
After this the training and test data is created and used to train the residual model.
Thereafter, the residual model is used to create the hybrid and the hybrid half trajectory
Finally, an image of all trajectories is created and saved in the optimal trajectory folder.
"""
import numpy as np
import pandas as pd
from visualisation import trajectory_plotting
import utils
from optimal_trajectory import calculate_optimal_trajectory
from optimal_trajectory.comparison import compare_different_N
import model_zoo
import simulate
from data_generation import simulation_data_generation
import path_handling
from NN_modeling import pytorch_NN
from matplotlib import pyplot as plt
from tqdm import trange


def _generate_physical_and_plant_trajectories():
    print('generating optimal physical and plant trajectories...')
    utils.blockPrint()
    calculate_optimal_trajectory.calculate_and_save_optimal_trajectory('physical_trajectory_free_Ts', 200,
                                                                       model_zoo.ERHARD_PHYSICAL_PARAMETER, False,
                                                                       'default initial guess for all trajectories'
                                                                       ' free Ts physical trajectory'
                                                                       ' generated using setup.py',
                                                                       visualize=False, use_initial_guess=False)
    calculate_optimal_trajectory.calculate_and_save_optimal_trajectory('physical_trajectory', 88,
                                                                       model_zoo.ERHARD_PHYSICAL_PARAMETER, True,
                                                                       'Default physical trajectory'
                                                                       ' generated using setup.py',
                                                                       visualize=False)

    calculate_optimal_trajectory.calculate_and_save_optimal_trajectory('plant_trajectory', 70,
                                                                       model_zoo.ERHARD_PLANT_PARAMETER, True,
                                                                       'Default plant trajectory'
                                                                       ' generated using setup.py',
                                                                       visualize=False)
    utils.enablePrint()


def _generate_training_and_test_data():
    # training Data
    sim_parameter = simulate.get_simulation_parameter(simulation_time=100)
    note = 'Auto generated simulation by setup.py the resulting data is used to train/test the NN'
    print('run simulations for training data...')
    s_values = np.linspace(0.1, 10, 20)
    pbar = trange(20, unit='Training_simulations')
    for i in pbar:
        s = s_values[i]
        utils.blockPrint()
        pf_control_parameter = simulate.get_control_parameter(controller_type='pf')
        pf_control_parameter['pf_parameter']['S'] = [[s, 0],
                                                     [0, 0]]
        tt_control_parameter = simulate.get_control_parameter(controller_type='tt')
        tt_control_parameter['tt_parameter']['S'] = s

        simulate.run_simulation(sim_parameter, pf_control_parameter, f'pf_simulation_{s}', 'training_data', note, False,
                                overwrite=False, dont_ask_user=True)
        simulate.run_simulation(sim_parameter, tt_control_parameter, f'tt_simulation{s}', 'training_data', note, False,
                                overwrite=False, dont_ask_user=True)
        utils.enablePrint()
    pbar.close()

    # test data
    sim_parameter['simulation_time'] = 120
    pf_control_parameter = simulate.get_control_parameter(controller_type='pf')
    tt_control_parameter = simulate.get_control_parameter(controller_type='tt')
    tt_control_parameter['reference'] = 'plant_trajectory'
    pf_control_parameter['reference'] = 'plant_trajectory'
    print('run simulations for test data...')
    utils.blockPrint()
    simulate.run_simulation(sim_parameter, tt_control_parameter, f'tt_simulation', 'test_data', note, False,
                            overwrite=False, dont_ask_user=True)
    simulate.run_simulation(sim_parameter, pf_control_parameter, f'pf_simulation', 'test_data', note, False,
                            overwrite=False, dont_ask_user=True)
    utils.enablePrint()


def _generate_residual_model():
    print('train residual model...')
    data = pd.read_csv(path_handling.TRAINING_DATA_DIR / 'autogenerated_simulation_data.csv')
    train = data.sample(frac=0.8, random_state=200)
    val = data.drop(train.index)
    pytorch_NN.create_residual_MLP_model('default_mlp', train, val)


def _calculate_hybrid_trajectories():
    print('find optimal period duration for hybrid trajectroy...')
    best_N = compare_different_N(60, 80, 'hybrid', 'compare_different_N_hybrid', 1, True, limit_delta_u=True)
    print('calculate optimal hybrid trajectory')
    utils.blockPrint()
    calculate_optimal_trajectory.calculate_and_save_optimal_trajectory('hybrid_trajectory', best_N,
                                                                       model_zoo.ERHARD_PHYSICAL_PARAMETER, True,
                                                                       'Default hybrid trajectory'
                                                                       ' generated using setup.py',
                                                                       visualize=False,
                                                                       residual_state_model_name='default_mlp',
                                                                       residual_thrust_model_name='default_mlp',
                                                                       use_stronger_constraints=True,
                                                                       limit_delta_u=True)
    utils.enablePrint()
    print('find optimal period duration for hybrid half trajectroy...')
    best_N = compare_different_N(30, 40, 'hybrid', 'compare_different_N_hybrid_half', 1, True, half=True,
                                 limit_delta_u=True)
    print('calculate optimal hybrid half trajectory')
    utils.blockPrint()
    calculate_optimal_trajectory.calculate_and_save_optimal_trajectory('hybrid_half_trajectory', best_N,
                                                                       model_zoo.ERHARD_PHYSICAL_PARAMETER, True,
                                                                       'Default hybrid half trajectory'
                                                                       ' generated using setup.py',
                                                                       visualize=False,
                                                                       residual_state_model_name='default_mlp',
                                                                       residual_thrust_model_name='default_mlp',
                                                                       use_stronger_constraints=True,
                                                                       calculate_half=True, limit_delta_u=True)
    utils.enablePrint()


def _make_figure_of_created_trajectories():
    trajectory_states = [utils.load_optimal_trajectories(trajectory_id).x for trajectory_id in
                         ['physical_trajectory', 'plant_trajectory', 'hybrid_trajectory', 'hybrid_half_trajectory']]
    labels = ['physical', 'plant', 'hybrid', 'hybrid_half']
    fig, ax = trajectory_plotting.plot_phi_theta_plane(trajectory_states,
                                                       labels=labels)
    fig.savefig(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'trajectory comparison.png')
    # plt.close(fig)

    trajectory_inputs = [utils.load_optimal_trajectories(trajectory_id).u for trajectory_id in
                         ['physical_trajectory', 'plant_trajectory', 'hybrid_trajectory', 'hybrid_half_trajectory']]
    trajectory_times = [utils.load_optimal_trajectories(trajectory_id).t for trajectory_id in
                        ['physical_trajectory', 'plant_trajectory', 'hybrid_trajectory', 'hybrid_half_trajectory']]
    fig2, axs2 = trajectory_plotting.plot_all_states_and_u(trajectory_states, u=trajectory_inputs, labels=labels,
                                                           t=trajectory_times)
    fig2.savefig(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'trajectory comparison time domain.png')
    plt.close(fig2)


def main():
    _generate_physical_and_plant_trajectories()
    _generate_training_and_test_data()
    simulation_data_generation.extract_data_from_simulation('autogenerated_simulation_data',
                                                            path_handling.TRAINING_DATA_DIR,
                                                            path_handling.SIMULATION_DIR / 'training_data')
    simulation_data_generation.extract_data_from_simulation('test',
                                                            path_handling.TEST_DATA_DIR,
                                                            path_handling.SIMULATION_DIR / 'test_data')
    _generate_residual_model()
    _calculate_hybrid_trajectories()
    _make_figure_of_created_trajectories()


if __name__ == '__main__':
    main()