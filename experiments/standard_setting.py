""" Run this script to run some simulations in the standard setting.

It first runs long simulations using the default control parameters.
Then, it compares different parameters and finally shows a trade-off between tracking accuracy and thrust.
This will take some time to run all the required simulations!
Use it also to see how to conduct your own simulations.
"""
import simulate
from matplotlib import pyplot as plt
import numpy as np
import path_handling


def run_simulations_with_default_parameters():
    simulation_parameter = simulate.get_simulation_parameter(1200)
    folders = []
    for pred_model_name in ['plant', 'physical', 'hybrid']:
        for controller_type in ['tt', 'pf']:
            save_folder = f'default_parameter/{pred_model_name}_prediction_model_{controller_type}'
            folders.append(save_folder)
            for reference in ['plant', 'physical', 'hybrid', 'hybrid_half']:
                print(f'Simulate: controller_type : {controller_type}, prediction model : {pred_model_name}, '
                      f'reference : {reference}')
                control_parameter = simulate.get_control_parameter(controller_type=controller_type,
                                                                   prediction_model_name=pred_model_name,
                                                                   reference_name=f'{reference}_trajectory')

                simulate.run_simulation(simulation_parameter, control_parameter, name=f'{reference}_trajectory',
                                        save_folder=save_folder,
                                        visualise=False,
                                        overwrite=False, dont_ask_user=True, notes='')

    simulate.make_table_of_simulations('default_parameter_experiments', folders)


def compare_different_parameter():
    simulation_parameter = simulate.get_simulation_parameter(60)
    pred_model_name = 'physical'
    s_values = np.concatenate((np.linspace(0, 1, 10), np.linspace(1.1, 10, 20)))
    q11_values = np.concatenate((np.linspace(10, 100, 10), np.linspace(110, 8000, 20)))
    s_table = []
    q11_table = []
    names_table = []

    for controller_type in ['tt', 'pf']:
        save_folder = f'parameter_comparison_{controller_type}'
        folders = [f'parameter_comparison_{controller_type}']
        for s in s_values:
            for q11 in q11_values:
                print(f'Simulate: controller_type : {controller_type}, prediction model : {pred_model_name}, '
                      f's : {s}, q11 : {q11}')
                control_parameter = simulate.get_control_parameter(controller_type=controller_type,
                                                                   prediction_model_name=pred_model_name,
                                                                   reference_name='physical_trajectory')
                if controller_type == 'pf':
                    parameter = control_parameter['pf_parameter']
                    parameter['Q'] = [[q11, 0, 0],
                                      [0, 100, 0],
                                      [0, 0, 100]]
                    parameter['S'] = [[s, 0],
                                      [0, 0]]
                else:
                    parameter = control_parameter['tt_parameter']
                    parameter['Q'] = [[q11, 0, 0],
                                      [0, 100, 0],
                                      [0, 0, 100]]
                    parameter['S'] = s
                s_table.append(s)
                q11_table.append(q11)
                names_table.append(f's_{s}_q11_{q11}')

                simulate.run_simulation(simulation_parameter, control_parameter, name=f's_{s}_q11_{q11}',
                                        save_folder=save_folder,
                                        visualise=False,
                                        overwrite=False, notes='', dont_ask_user=True)

        extra_colums = {'s': s_table, 'q11': q11_table, 'name': names_table}

        table = simulate.make_table_of_simulations(f'compare_different_parameter_{controller_type}', folders,
                                                   extra_colums=extra_colums)
        table = table.sort_values(by=['s', 'q11'])
        X = np.array(table['s']).reshape(len(s_values), -1)
        Y = np.array(table['q11']).reshape(len(s_values), -1)

        for Z_name in ['average_thrust', 'average_tacking_error_in_m', 'min_height']:
            Z = np.array(table[Z_name]).reshape(len(s_values), -1)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=2, cstride=4,
                            alpha=0.3)

            thrust_lim = [table[Z_name].min(), table[Z_name].max()]
            s_lim = [table['s'].min(), table['s'].max()]
            q_lim = [table['q11'].min(), table['q11'].max()]
            # Plot projections of the contours for each dimension.  By choosing offsets
            # that match the appropriate axes limits, the projected contours will sit on
            # the 'walls' of the graph.
            #  ax.contour(X, Y, Z, zdir='z', offset=thrust_lim[0], cmap='coolwarm')
            ax.contour(X, Y, Z, zdir='x', offset=s_lim[0], cmap='coolwarm')
            ax.contour(X, Y, Z, zdir='y', offset=q_lim[1], cmap='coolwarm')

            ax.set(xlim=s_lim, ylim=q_lim, zlim=thrust_lim,
                   xlabel='s', ylabel='q11', zlabel=Z_name)
            fig.savefig(path_handling.SIMULATION_DIR / f'parameter_comparison_{Z_name}_{controller_type}.png')
            plt.close(fig)


def show_thrust_accuracy_tradeoff():
    simulation_parameter = simulate.get_simulation_parameter(60)
    folders = []
    pred_model_name = 'physical'
    s_values = np.concatenate((np.linspace(0, 1, 20), np.linspace(1.1, 10, 20)))

    s_table = []
    names_table = []

    save_folder = f'thrust_accuracy_tradeoff'
    folders.append(save_folder)
    for i, s in enumerate(s_values):
        control_parameter = simulate.get_control_parameter(controller_type='pf',
                                                           prediction_model_name=pred_model_name,
                                                           reference_name='physical_trajectory')
        pf_parameter = control_parameter['pf_parameter']
        pf_parameter['S'] = [[s, 0],
                             [0, 0]]
        s_table.append(s)
        name = f'number_{i}_s_{s}'
        names_table.append(name)

        simulate.run_simulation(simulation_parameter, control_parameter, name=name,
                                save_folder=save_folder,
                                visualise=False,
                                overwrite=False, dont_ask_user=True, notes='')

    extra_colums = {'s': s_table, 'name': names_table}

    table = simulate.make_table_of_simulations('thrust_accuracy_tradeoff', folders, extra_colums=extra_colums)
    table = table.sort_values(by=['s'])
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(table['s'], table['average_thrust'], 'g-')
    ax2.plot(table['s'], table['average_tacking_error_in_m'], 'b-')

    ax1.set_xlabel('s')
    ax1.set_ylabel('average_thrust', color='g')
    ax2.set_ylabel('average_tacking_error_in_m', color='b')

    fig.savefig(path_handling.SIMULATION_DIR / 'thrust_accuracy_tradeoff.png')


def main():
    run_simulations_with_default_parameters()
    compare_different_parameter()
    show_thrust_accuracy_tradeoff()


if __name__ == '__main__':
    main()
    plt.show()
