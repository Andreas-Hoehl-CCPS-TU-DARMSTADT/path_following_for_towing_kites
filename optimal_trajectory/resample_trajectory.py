import numpy as np
import utils
from visualisation import trajectory_plotting as myplot
from scipy.interpolate import CubicSpline
import model_zoo


def create_resampled_trajectory(name: str, N_new: int):
    """resamples an existing trajectory

    Uses Cubic splines to resample a trajectory. It then saves this resampled trajectory with the name:
    f"{name}_resampled_to_{N_new}" in the $OPTIMAL_TRAJECTORY_DIR to make it available for further simulations.

    Parameters
    ----------
    name : str
        name of the original trajectory
    N_new : int
        number of sampling point of the new trajectory

    """
    trajectory = utils.load_optimal_trajectories(name)

    new_t_final = N_new * model_zoo.ERHARD_Ts
    new_t = np.arange(len(trajectory.x)) / (len(trajectory.x) - 1) * new_t_final

    x_spline = CubicSpline(new_t, trajectory.x, bc_type='periodic')
    u_spline = CubicSpline(new_t, np.concatenate((trajectory.u.flatten(), trajectory.u[0])).reshape(-1, 1),
                           bc_type='periodic')
    np.append(trajectory.integrated_thrust_per_time_step, trajectory.integrated_thrust_per_time_step[0])
    thrust_spline = CubicSpline(new_t, np.append(trajectory.integrated_thrust_per_time_step,
                                                 trajectory.integrated_thrust_per_time_step[0]).reshape(-1, 1),
                                bc_type='periodic')

    new_t_Ts = np.arange(N_new + 1) * 0.27
    new_x = x_spline(new_t_Ts)
    new_u = u_spline(new_t_Ts[:-1])

    # myplot.plot_all_states_and_u(x=[trajectory.x, new_x], u=[trajectory.u, new_u], t=[trajectory.t, new_t_Ts])
    # myplot.plot_phi_theta_plane(x=[trajectory.x, new_x])

    new_trajectory = utils.OptimalTrajectory(f"{name}_resampled_to_{N_new}", new_x, new_u, new_t_final,
                                             thrust_spline(new_t_Ts[:-1]), None)
    new_trajectory.save()
