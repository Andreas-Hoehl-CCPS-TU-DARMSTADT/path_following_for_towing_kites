""" This script reproduces the results for the trajectory tracking controller presented in the paper

Run standard_setting.py first to create the simulations which are loaded in this script.
to avoid an overload of figures most lines are commented. Uncomment to see the desired results.
"""
from simulate import visualize_result
import simulate
from optimal_trajectory.resample_trajectory import create_resampled_trajectory


def _simulate_tt_with_resampled_trajectory(N):
    create_resampled_trajectory('physical_trajectory', N)
    simulation_parameter = simulate.get_simulation_parameter(240)
    control_parameter = simulate.get_control_parameter('tt', prediction_model_name='physical',
                                                       reference_name=f'physical_trajectory_resampled_to_{N}')
    simulate.run_simulation(simulation_parameter, control_parameter,
                            f'physical_trajectory_resampled_to_{N}_physical_prediction_model', 'resampled_trajectories',
                            notes='', visualise=False, overwrite=False, dont_ask_user=True)
    visualize_result(f'physical_trajectory_resampled_to_{N}_physical_prediction_model', 'resampled_trajectories')


def main():
    # physical trajectory and physical prediction model
    visualize_result('physical_trajectory', 'default_parameter/physical_prediction_model_tt')

    # physical trajectory and plant prediction model
    visualize_result('physical_trajectory', 'default_parameter/plant_prediction_model_tt')

    # resampled physical trajectory and physical prediction model
    _simulate_tt_with_resampled_trajectory(65)
    _simulate_tt_with_resampled_trajectory(70)
    _simulate_tt_with_resampled_trajectory(73)
    _simulate_tt_with_resampled_trajectory(88)


if __name__ == '__main__':
    main()
