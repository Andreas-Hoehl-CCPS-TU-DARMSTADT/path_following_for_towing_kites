""" Calculates the optimal trajectories for the different prediction models and different period durations.

"""
import matplotlib.pyplot as plt
from optimal_trajectory.comparison import compare_different_N
import utils
from visualisation import trajectory_plotting


def _calculate_optimal_trajectories_for_different_N():
    # already calculated in setup.py
    # compare_different_N(60, 80, 'hybrid', 'compare_different_N_hybrid', 1, True, limit_delta_u=True)
    # compare_different_N(30, 40, 'hybrid', 'compare_different_N_hybrid_half_trivial', 1, False, half=True,
    #                     limit_delta_u=False)
    compare_different_N(60, 80, 'hybrid', 'compare_different_N_hybrid_additional_constraints_no_init', 1, True,
                        half=False, limit_delta_u=True, use_initial_guess=False)
    # compare_different_N(80, 100, 'physical', 'compare_different_N_physical', 3, False)
    # compare_different_N(60, 80, 'plant', 'compare_different_N_plant', 3, False)


def _plot_reference_trajectories():
    physical_trajectory = utils.load_optimal_trajectories('physical_trajectory')
    plant_trajectory = utils.load_optimal_trajectories('plant_trajectory')
    hybrid_trajectory = utils.load_optimal_trajectories('hybrid_trajectory')
    hybrid_half_trajectory = utils.load_optimal_trajectories('hybrid_half_trajectory')
    trajectories = [plant_trajectory, physical_trajectory, hybrid_trajectory, hybrid_half_trajectory]
    trajectory_plotting.plot_phi_theta_plane(
        [trajectory.x for trajectory in trajectories],
        labels=['plant', 'physical', 'hybrid', 'hybrid_half'])
    trajectory_plotting.plot_all_states_and_u([trajectory.x for trajectory in trajectories],
                                              u=[trajectory.u for trajectory in trajectories],
                                              thrust=[trajectory.integrated_thrust_per_time_step for trajectory in
                                                      trajectories],
                                              labels=['plant', 'physical', 'hybrid', 'hybrid_half'],
                                              t=[trajectory.t for trajectory in trajectories])


def main():
    _calculate_optimal_trajectories_for_different_N()
    # _plot_reference_trajectories()
    plt.show()


if __name__ == '__main__':
    main()
