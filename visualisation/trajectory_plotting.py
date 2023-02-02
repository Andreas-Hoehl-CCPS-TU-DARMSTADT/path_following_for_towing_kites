""" This module provides plotting functions.

"""
import numpy as np
import model_zoo
import path_handling
import utils
from matplotlib import pyplot as plt
from typing import List, Tuple, Union

rad_to_deg = 1 / np.pi * 180


def plot_phi_theta_plane(x: Union[np.ndarray, List[np.ndarray]], colors: List[str] = None,
                         labels: List[str] = None, title: str = r'$\phi$-$\theta$-plane',
                         plot_arrows: bool = False, fig_ax=None) -> Tuple[plt.Figure, plt.Axes]:
    """ plots the phi theta plane of a given trajectory.

    :param x: state trajectory (or list of trajectories) row wise (col0 = theta, col1 = phi)
    :param colors: colors of the trajectories
    :param labels: labels of the trajectories
    :param title: title of the plot
    :param plot_arrows: plots the direction of the curve using arrows
    :return: figure and ax of the plot
    """

    if fig_ax is None:
        fig, ax = utils.get_ax()
    else:
        fig, ax = fig_ax
    if not isinstance(x, list):
        x = [x]
        print_legend = False
    else:
        print_legend = True

    if colors is None:
        colors = list(path_handling.TU_COLORS.values())

    if labels is None:
        labels = [''] * len(x)

    for i, trajectory in enumerate(x):
        ax.plot(trajectory[0, 1] * rad_to_deg, trajectory[0, 0] * rad_to_deg, 'o', color=colors[i % len(colors)],
                markersize=8)
        ax.plot(trajectory[:, 1] * rad_to_deg, trajectory[:, 0] * rad_to_deg, colors[i % len(colors)], label=labels[i])
        if plot_arrows:
            phi1 = np.ma.masked_array(trajectory[:, 1][:-1], np.diff(trajectory[:, 1]) >= 0)
            phi2 = np.ma.masked_array(trajectory[:, 1][:-1], np.diff(trajectory[:, 1]) <= 0)
            ax.plot(phi1 * rad_to_deg, trajectory[:, 0][:-1] * rad_to_deg, '>', color=colors[i % len(colors)],
                    markersize=4)
            ax.plot(phi2 * rad_to_deg, trajectory[:, 0][:-1] * rad_to_deg, '<', color=colors[i % len(colors)],
                    markersize=4)
        ax.plot(trajectory[-1, 1] * rad_to_deg, trajectory[-1, 0] * rad_to_deg, 'x', color=colors[i % len(colors)],
                markersize=8)

    ax.grid(True)
    ax.set_xlabel(r'$\phi$ in $[\deg]$')
    ax.set_ylabel(r'$\theta$ in $[\deg]$')
    ax.set_title(title)
    if print_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    return fig, ax


def plot_all_states_and_u(x: Union[np.ndarray, List[np.ndarray]], Ts: float = model_zoo.ERHARD_Ts,
                          t: Union[np.ndarray, List[np.ndarray]] = None,
                          u: Union[np.ndarray, List[np.ndarray]] = None,
                          thrust: Union[np.ndarray, List[np.ndarray]] = None, colors: List[str] = None,
                          labels: List[str] = None, title=r'states') -> Tuple[plt.Figure, np.ndarray]:
    """ plots als states and u over the time.

    :param x: state trajectory (or list of trajectories) row wise (col0 = theta, col1 = phi)
    :param Ts: sampling time (to scale t axis)
    :param t: time vector or list of time vectors (Ts is ignored if t is given)
    :param u: input trajectory (or list of trajectories)
    :param thrust: thrust values
    :param colors: colors of the trajectories
    :param labels: labels of the trajectories
    :param title: title of the plot
    :return: figure and ax of the plot
    """

    draw_u = u is not None
    draw_thrust = thrust is not None

    if not isinstance(x, list):
        x = [x]
        u = [u]
        thrust = [thrust]
        print_legend = False
    else:
        print_legend = True

    if t is None:
        t = [np.arange(len(x[0])) * Ts] * len(x)
    elif not isinstance(t, list):
        t = [t]

    if colors is None:
        colors = list(path_handling.TU_COLORS.values())

    if labels is None:
        labels = [''] * len(x)

    n = x[0].shape[1]
    if draw_u:
        n += 1
    if draw_thrust:
        n += 1

    fig, axs = utils.get_ax(n_rows=n, sharex=True)

    for i in range(x[0].shape[1]):
        for j in range(len(x)):
            if i < 3:
                axs[i].plot(t[j], x[j][:, i] * rad_to_deg, label=labels[j], color=colors[j % len(colors)])
            else:
                axs[i].plot(t[j], x[j][:, i], label=labels[j], color=colors[j % len(colors)])
        axs[i].grid()
        axs[i].set_ylabel(['theta', 'phi', 'psi', 'l'][i])

    if draw_thrust:
        for j in range(len(thrust)):
            axs[x[0].shape[1]].plot(t[j][:-1], thrust[j], label=labels[j], color=colors[j % len(colors)])
        axs[x[0].shape[1]].grid()
        axs[x[0].shape[1]].set_ylabel('thrust')

    if draw_u:
        for j in range(len(u)):
            axs[-1].step(t[j][:-1], u[j], label=labels[j], color=colors[j % len(colors)], where='post')
        axs[-1].grid()
        axs[-1].set_xlabel('t in s')
        axs[-1].set_ylabel('u')
        axs[-1].set_title('control output')
        axs[-1].set_xlabel('t in s')
    fig.suptitle(title)
    if print_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    return fig, axs


def plot_x_y_z_trajectory(x, L=400, colors=None, labels=None):
    # quarter circle for background
    phi = np.linspace(-np.pi / 2, np.pi / 2, 100).reshape(-1, 1)
    theta = np.linspace(0, np.pi / 2, 100)

    x_c = (L * np.cos(theta) * np.ones(shape=phi.shape)).flatten()
    y_c = (L * np.sin(theta) * np.sin(phi)).flatten()
    z_c = (L * np.sin(theta) * np.cos(phi)).flatten()

    # height constrain
    phi = np.linspace(-np.pi / 2 + 0.3, np.pi / 2 - 0.3, 100)
    theta = np.arcsin(100 / (L * np.cos(phi)))
    x_h = (L * np.cos(theta))
    y_h = (L * np.sin(theta) * np.sin(phi))
    z_h = (L * np.sin(theta) * np.cos(phi))

    fig, ax = utils.get_ax(dim3=True)
    ax.plot_trisurf(x_c, y_c, z_c, linewidth=0.2, antialiased=True, alpha=0.5, color='grey')
    ax.plot(x_h, y_h, z_h, 'black')

    # trajectories
    if not isinstance(x, list):
        x = [x]
        print_legend = False
    else:
        print_legend = True

    if colors is None:
        colors = list(path_handling.TU_COLORS.values())
    if labels is None:
        labels = [''] * len(x)
    for i, trajectory in enumerate(x):
        x_cart = utils.transform_to_cartesian(trajectory, L)

        ax.plot(x_cart[:, 0], x_cart[:, 1], x_cart[:, 2], color=colors[i], label=labels[i])

    ax.view_init(15, 190)
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim([0, 400])
    ax.set_xlim([0, 400])
    ax.set_ylim([-400, 400])
    ax.set_title('3_d view')
    if print_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return fig
