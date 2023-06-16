""" In this example we investigate the maximum thrust for different minimal heights.

"""
import numpy as np
from optimal_trajectory.calculate_optimal_trajectory import calculate_and_save_optimal_trajectory
import model_zoo
from matplotlib import pyplot as plt
import pandas as pd
import path_handling
from scipy.interpolate import CubicSpline
import simulate


def _compute_max_thrust_for_different_heights():
    minimal_heights = np.linspace(1, 399, 100)
    thrust = []
    for minimal_height in minimal_heights:
        name = f'thrust_height_tradeoff_h_{minimal_height}'
        trajectory = calculate_and_save_optimal_trajectory(name, 200,
                                                           model_zoo.ERHARD_PLANT_PARAMETER, False,
                                                           '', visualize=False, use_initial_guess=True,
                                                           hmin=minimal_height,
                                                           sub_folder_name='thrust_height_tradeoff')
        thrust.append(trajectory.thrust_integral)

    fig = plt.figure()
    plt.plot(minimal_heights, thrust)
    plt.xlabel('h min')
    plt.ylabel('thrust')
    plt.grid()
    fig.savefig(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'thrust_height_tradeoff.png')
    thrust_df = pd.DataFrame({
        'minimal_height': minimal_heights,
        'average_thrust': thrust
    })

    # save also as txt in order to use pgf plots in latex
    with open(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'thrust_height_tradeoff.txt', 'w') as f:
        dfAsString = thrust_df.to_string(header=True, index=False)
        f.write(dfAsString)
    plt.show()
    plt.close(fig)
    thrust_and_height_array = np.array([minimal_heights, thrust]).transpose()
    np.save(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'thrust_height_tradeoff.npy', thrust_and_height_array)


def _give_percentage_of_max_thrust(thrust_values, minimal_height_values):
    thrust_and_height = np.load(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'thrust_height_tradeoff.npy')
    thrust_spline = CubicSpline(x=thrust_and_height[:, 0], y=thrust_and_height[:, 1])
    for thrust, height in zip(thrust_values, minimal_height_values):
        best_thrust = thrust_spline(height)
        percentage = 100 - (best_thrust - thrust) / best_thrust * 100
        print(
            f'an average thrust of {thrust} with a minimal height of {height} corresponds to '
            f'{percentage}% of the maximum achievable thrust')


def _look_at_percentage_for_different_parameter_settings():
    s_values = list(np.logspace(-4, np.log10(8), 40))
    q11_values = [10, 50, 100, 500, 1000, 5000]
    s_table = []
    q11_table = []
    names_table = []

    for controller_type in ['pf', 'tt']:
        folders = [f'parameter_comparison_{controller_type}']
        for s in s_values:
            for q11 in q11_values:
                s_table.append(s)
                q11_table.append(q11)
                names_table.append(f's_{s}_q11_{q11}')

        extra_colums = {'s': s_table, 'q11': q11_table, 'name': names_table}
        table = simulate.make_table_of_simulations(f'compare_different_parameter_{controller_type}', folders,
                                                   extra_colums=extra_colums)
        table = table.sort_values(by=['s', 'q11'])
        thrust_and_height = np.load(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'thrust_height_tradeoff.npy')
        thrust_spline = CubicSpline(x=thrust_and_height[:, 0], y=thrust_and_height[:, 1])
        average_thrust = table['average_thrust'].to_numpy()
        average_thrust_not_corrected = table['average_thrust_not_corrected'].to_numpy()
        index_where_thrust_is_wrong = np.abs(((table['average_thrust_not_corrected'] - average_thrust) / table[
            'average_thrust_not_corrected']).to_numpy()) > 0.05
        average_thrust[index_where_thrust_is_wrong] = average_thrust_not_corrected[index_where_thrust_is_wrong]
        percentage = 100 - (thrust_spline(table['min_height']) - average_thrust) / thrust_spline(
            table['min_height']) * 100
        X = np.array(table['s']).reshape(len(s_values), -1)
        Y = np.array(table['q11']).reshape(len(s_values), -1)

        Z = np.array(percentage).reshape(len(s_values), -1)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=2, cstride=4,
                        alpha=0.3)

        thrust_lim = [percentage.min(), percentage.max()]
        s_lim = [table['s'].min(), table['s'].max()]
        q_lim = [table['q11'].min(), table['q11'].max()]
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        #  ax.contour(X, Y, Z, zdir='z', offset=thrust_lim[0], cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='x', offset=s_lim[0], cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='y', offset=q_lim[1], cmap='coolwarm')

        ax.set(xlim=s_lim, ylim=q_lim, zlim=thrust_lim,
               xlabel='s', ylabel='q11', zlabel='percentage')
        fig.savefig(path_handling.SIMULATION_DIR / f'parameter_comparison_percentage_{controller_type}.png')
        plt.close(fig)

        for i in range(len(q11_values)):
            q11_i = Y[0, i]
            s_i = X[:, 0]
            percentage_i = Z[:, i]
            average_thrust_i = np.array(table['average_thrust']).reshape(len(s_values), -1)[:, i]
            min_height_i = np.array(table['min_height']).reshape(len(s_values), -1)[:, i]
            error_i = np.array(table['average_tacking_error_in_m']).reshape(len(s_values), -1)[:, i]
            mydf = pd.DataFrame({
                's': s_i,
                'percentage': percentage_i,
                'min_height': min_height_i,
                'error': error_i,
                'thrust': average_thrust_i
            })
            with open(path_handling.OPTIMAL_TRAJECTORIES_DIR / f'different_parameter_{controller_type}_{q11_i}.txt',
                      'w') as f:
                dfAsString = mydf.to_string(header=True, index=False)
                f.write(dfAsString)


if __name__ == '__main__':
    # _compute_max_thrust_for_different_heights()
    _give_percentage_of_max_thrust(np.array([536.1, 554.4, 542.3, 570.0, 559.5, 565.0, 570.0]) * 1000,
                                   [94.0, 95.0, 95.6, 94.2, 99.4, 99.5, 98.4])
    _look_at_percentage_for_different_parameter_settings()
