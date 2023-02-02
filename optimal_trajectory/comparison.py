import model_zoo
from optimal_trajectory.calculate_optimal_trajectory import calculate_and_save_optimal_trajectory
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import path_handling
from tqdm import trange
import utils


def compare_different_N(N_low, N_high, prediction_model_name, sub_folder_name, number_of_steps,
                        use_stronger_constraints, half=False, limit_delta_u=False, visualize=False,
                        use_initial_guess=True):
    """ calculates and compares the optimal trajectories for different period durations T=T_s*N

    Parameters
    ----------
    N_low : int
        the lowest value for N
    N_high : int
        the highest value for N
    prediction_model_name : str
        name of the prediction model: either 'plant', 'hybrid', 'physical'
    sub_folder_name : str
        name of the save folder within $OPTIMAL_TRAJECTORY_DIR
    number_of_steps : int
        number of rk4 steps used in the discretization
    use_stronger_constraints : bool
        whether the additional constraints (see paper) should be used
    half : bool
        whether only half of the trajectory should be calculated
    limit_delta_u : bool
        whether delta u should be limited as a constraint
    visualize : bool
        whether the result should be visualized
    use_initial_guess:
        whether the optimal physical trajectory should be used as initial guess

    Returns
    -------
    int
        value of N for which the optimal achieved the highest average thrust
    """
    if prediction_model_name == 'physical' or prediction_model_name == 'hybrid':
        parameter = model_zoo.ERHARD_PHYSICAL_PARAMETER
    elif prediction_model_name == 'plant':
        parameter = model_zoo.ERHARD_PLANT_PARAMETER
    else:
        raise RuntimeError('unknown prediction model name')

    if prediction_model_name == 'hybrid':
        residual_model_name = 'default_mlp'
    else:
        residual_model_name = ''
    thrust_values = []
    Ns = list(range(N_low, N_high + 1))
    pbar = trange(len(Ns), unit='Optimal Trajectories')
    for i in pbar:
        N = Ns[i]
        utils.blockPrint()
        trajectory = calculate_and_save_optimal_trajectory(f'trajectory{N}', N, parameter,
                                                           visualize=visualize,
                                                           sub_folder_name=sub_folder_name,
                                                           use_initial_guess=use_initial_guess,
                                                           number_of_steps=number_of_steps,
                                                           use_stronger_constraints=use_stronger_constraints,
                                                           residual_state_model_name=residual_model_name,
                                                           residual_thrust_model_name=residual_model_name,
                                                           calculate_half=half,
                                                           limit_delta_u=limit_delta_u)
        thrust_values.append(trajectory.thrust_integral)
        utils.enablePrint()
    pbar.close()

    fig = plt.figure()
    plt.plot(Ns, thrust_values)
    plt.grid()
    fig.savefig(path_handling.OPTIMAL_TRAJECTORIES_DIR / sub_folder_name / 'thrust_comparison.png')
    thrust_df = pd.DataFrame({'Ns': Ns, 'thrust': thrust_values})
    thrust_df.to_csv(path_handling.OPTIMAL_TRAJECTORIES_DIR / sub_folder_name / 'thrust_comparison.csv')
    print(
        f'the highest thrust value is {np.max(thrust_values)} and was achieved for N = {Ns[np.argmax(thrust_values)]}')
    if visualize:
        plt.show()
    plt.close(fig)
    return Ns[np.argmax(thrust_values)]
