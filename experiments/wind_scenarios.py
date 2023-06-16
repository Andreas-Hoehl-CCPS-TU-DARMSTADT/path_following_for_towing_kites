""" This file runs closed-loop simulations for different wind scenarios

"""
import simulate
import numpy as np
from matplotlib import pyplot as plt
import path_handling


def _run_harmonic_wind_simulations(controller_type):
    f_table = []
    m_table = []
    names_table = []

    frequencies = list(np.logspace(-2, np.log10(.5), 50))
    magnitudes = [2, 4, 6, 8, 10]

    folders = [f'harmonic_wind_{controller_type}']

    control_parameter = simulate.get_control_parameter(controller_type=controller_type,
                                                       prediction_model_name='physical',
                                                       reference_name='physical_trajectory')

    simulation_parameter = simulate.get_simulation_parameter(200)
    simulation_parameter['wind_function_index'] = 1

    for freq in frequencies:
        for mag in magnitudes:
            simulation_parameter['wind_frequency'] = freq
            simulation_parameter['magnitude'] = mag

            name = f'freq_{freq}_mag_{mag}'
            names_table.append(name)
            f_table.append(freq)
            m_table.append(mag)

            simulate.run_simulation(simulation_parameter, control_parameter, name=name,
                                    save_folder=folders[0],
                                    visualise=False,
                                    overwrite=False, dont_ask_user=True, notes='')

    extra_colums = {'freq': f_table, 'mag': m_table, 'name': names_table}

    table = simulate.make_table_of_simulations(f'harmonic_wind_{controller_type}', folders, extra_colums=extra_colums)
    table = table.sort_values(by=['freq', 'mag'])
    X = np.array(table['freq']).reshape(len(frequencies), -1)
    Y = np.array(table['mag']).reshape(len(frequencies), -1)

    for Z_name in ['success_rate', 'average_tacking_error_in_m', 'min_height']:
        Z = np.array(table[Z_name]).reshape(len(frequencies), -1)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=2, cstride=4,
                        alpha=0.3)

        thrust_lim = [table[Z_name].min(), table[Z_name].max()]
        freq_lim = [table['freq'].min(), table['freq'].max()]
        mag_lim = [table['mag'].min(), table['mag'].max()]
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        #  ax.contour(X, Y, Z, zdir='z', offset=thrust_lim[0], cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='x', offset=freq_lim[0], cmap='coolwarm')
        ax.contour(X, Y, Z, zdir='y', offset=mag_lim[1], cmap='coolwarm')

        ax.set(xlim=freq_lim, ylim=mag_lim, zlim=thrust_lim,
               xlabel='freq', ylabel='mag', zlabel=Z_name)
        fig.savefig(path_handling.SIMULATION_DIR / f'harmonic_wind_{Z_name}_{controller_type}.png')
        plt.close(fig)

    for mag in magnitudes:
        table_part = table[table['mag'] == mag][['freq', 'average_tacking_error_in_m']]
        with open(path_handling.SIMULATION_DIR / f'harmonic_wind_{controller_type}_{mag}.txt', 'w') as f:
            dfAsString = table_part.to_string(header=True, index=False)
            f.write(dfAsString)


def _run_wind_gust_simulation(controller_type):
    m_table = []
    names_table = []

    delay = 30
    magnitudes = [2, 4, 6, 8, 10]

    folders = [f'wind_gust_{controller_type}']

    control_parameter = simulate.get_control_parameter(controller_type=controller_type,
                                                       prediction_model_name='physical',
                                                       reference_name='physical_trajectory')

    simulation_parameter = simulate.get_simulation_parameter(80)
    simulation_parameter['wind_function_index'] = 2

    for mag in magnitudes:
        simulation_parameter['delay'] = delay
        simulation_parameter['magnitude'] = mag

        name = f'mag_{mag}'
        names_table.append(name)
        m_table.append(mag)

        simulate.run_simulation(simulation_parameter, control_parameter, name=name,
                                save_folder=folders[0],
                                visualise=False,
                                overwrite=False, dont_ask_user=True, notes='')

    extra_colums = {'mag': m_table, 'name': names_table}

    simulate.make_table_of_simulations(f'wind_gust_{controller_type}', folders, extra_colums=extra_colums)


def main():
    # _run_harmonic_wind_simulations('pf')
    # _run_harmonic_wind_simulations('tt')
    _run_wind_gust_simulation('pf')
    # _run_wind_gust_simulation('tt')


if __name__ == '__main__':
    main()
