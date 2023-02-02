""" This module provides helper functions.

"""
from __future__ import annotations
import path_handling
from matplotlib import pyplot as plt
import casadi as ca
import numpy as np
from typing import Callable, TypedDict, List, Tuple, Union, Dict, Optional
import pickle
from typing import TYPE_CHECKING
from abc import abstractmethod
from scipy.interpolate import CubicSpline
import pandas as pd
import json
from interfaces import SimulationResult
import sys, os

if TYPE_CHECKING:
    from model_zoo import PredictionModel


def get_ax(n_rows: int = 1, n_cols: int = 1, dim3: bool = False, sharex: bool = False, sharey: bool = False,
           fig_size: Tuple[int, int] = None, tight: bool = True) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    generates a figure object with the specified axis
    :param n_rows: number of rows of the subplots
    :param n_cols: number of columns of the subplots
    :param dim3: whether the plots should be 3 dimensional
    :param sharex: whether the x-axis of the subplots should be shared
    :param sharey: whether the y-axis of the subplots should be shared
    :param fig_size: tuple containing figure width and height
    :param tight: whether tight layout should be used
    :return: tuple of the figure and the axis or an array containing all axis
    """
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')  # 3d plots use this for each axis
    plt.rc('ytick', labelsize='x-small')
    # plt.rc('text', usetex=True)  # uncomment for final plots

    if fig_size is None:
        fig_size = (4 * n_cols, 3 * n_rows)
    if dim3:
        fig = plt.figure(figsize=fig_size)
        if n_rows != 1 or n_cols != 1:
            ax = np.array(
                [[fig.add_subplot(n_rows, n_cols, n_rows * (i - 1) + j, projection='3d') for i in range(1, n_cols + 1)]
                 for j in range(1, n_rows + 1)])

        else:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        # noinspection PyTypeChecker
        fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharey=sharey, sharex=sharex, tight_layout=tight)

    return fig, ax


def get_solution_state_trajectories_from_result(result: SimulationResult) -> np.ndarray:
    """fetches the solution state trajectory form the given result

    Parameters
    ----------
    result : SimulationResult
        the simulation result where MPC was used as controller.

    Returns
    -------
    np.ndarray
        three dimensional array of size (n,N,nx) where n is the number of control steps,
        N is the prediction horizon and nx is the state dimension
    """
    return np.array([result['info'][k]['x_sol'] for k in range(len(result['info']))])


class OptimalTrajectory:
    def __init__(self, name: str, x: np.ndarray, u: np.ndarray, t_final: float,
                 integrated_thrust_per_time_step: np.ndarray, doc=None):
        """
        class to store an optimal trajectory.
        :param x: o optimal
        :param u: u optimal
        :param t_final: length of the trajectory
        :param integrated_thrust_per_time_step: integrated thrust values for each time step
        :param doc: documentation dict of the trajectory
        """
        if doc is None:
            doc = {}
        self.x = x
        self.u = u
        self.Ts = t_final / (x.shape[0] - 1)
        self.t_final = t_final
        self.t = np.arange(x.shape[0]) / (x.shape[0] - 1) * self.t_final
        self.integrated_thrust_per_time_step = integrated_thrust_per_time_step
        self.thrust_integral = integrated_thrust_per_time_step.sum()
        self.doc = doc
        self.name = name

    def save(self, sub_folder_name='') -> None:
        """
        saves the Trajectory to disk
        """
        if sub_folder_name:
            path = path_handling.OPTIMAL_TRAJECTORIES_DIR / sub_folder_name / self.name
        else:
            path = path_handling.OPTIMAL_TRAJECTORIES_DIR / self.name

        if path.exists():
            pass
            # warnings.warn(f"You just overrode the optimal trajectory called {self.name}")
        else:
            os.makedirs(path)

        with open(path / 'trajectory.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        trajectory_df = pd.DataFrame({
            't': self.t,
            'integrated_thrust_per_time_step': np.append(self.integrated_thrust_per_time_step, np.nan),
            'u': np.append(self.u.flatten(), np.nan),
            'theta': self.x[:, 0],
            'phi': self.x[:, 1],
            'psi': self.x[:, 2]
        })

        # save also as txt in order to use pgf plots in latex
        with open(path / f'trajectory.txt', 'w') as f:
            dfAsString = trajectory_df.to_string(header=True, index=False)
            f.write(dfAsString)

        with open(path / 'doc.json', 'w') as output:
            json.dump(self.doc, output)

        # save image to see results faster
        from visualisation import trajectory_plotting  # quick fix to avoid circular import
        fig, ax = trajectory_plotting.plot_phi_theta_plane(self.x)
        fig.savefig(path.parent / f'{self.name}.png')
        plt.close(fig)


def get_random_initial_value(L: float, h_min: float) -> List:
    """
    generates a random initial value that satisfies the given minimum height
    :param L: tether length
    :param h_min: minimum height
    :return: initial value as list
    """
    lb = np.array([0, -np.pi / 2, -np.pi])
    ub = np.array([np.pi / 2, np.pi / 2, np.pi])
    for i in range(99999):
        x0 = np.random.random(3) * (ub - lb) + lb
        if h_min < L * np.sin(x0[0]) * np.cos(x0[1]):
            break
    else:
        raise RuntimeError('could not find feasible initial value')
    return list(x0)


def load_optimal_trajectories(trajectory_id: str, sub_folder_name: str = '') -> OptimalTrajectory:
    """
    loads an optimal Trajectory
    :param trajectory_id: name of the trajectory
    :param sub_folder_name: additional sub folder
    :return: optimal trajectory
    """

    class CustomUnpickler(pickle.Unpickler):

        def find_class(self, module, class_name):
            if class_name == 'OptimalTrajectory':
                return OptimalTrajectory
            return super().find_class(module, class_name)

    if sub_folder_name:
        path = path_handling.OPTIMAL_TRAJECTORIES_DIR / sub_folder_name / trajectory_id
    else:
        path = path_handling.OPTIMAL_TRAJECTORIES_DIR / trajectory_id

    trajectory = CustomUnpickler(
        open(path / ('trajectory' + '.pkl'), 'rb')).load()

    return trajectory


def create_poly_matrices(n_deg: int, collocation_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ creates the polynomial collocation matrices.

    :param n_deg: oder of the polynomial
    :param collocation_type: method to select collocation points
    :return: B, C, D
            where B contains the integral of the basis polynomials,
            C contains the values of the derivatives of the basis polynomials at the collocation points
            and D contains the values at the end of the interval of the basis polynomials.
    """
    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(n_deg, collocation_type))

    # Coefficients of the collocation equation
    C = np.zeros((n_deg + 1, n_deg + 1))

    # Coefficients of the continuity equation
    D = np.zeros(n_deg + 1)

    # Coefficients of the quadrature function
    B = np.zeros(n_deg + 1)

    # Construct polynomial basis
    for j in range(n_deg + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(n_deg + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the
        # continuity equation
        pder = np.polyder(p)
        for r in range(n_deg + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return B, C, D


def transform_to_cartesian(x: np.ndarray, L: float) -> np.ndarray:
    """
    transforms the erhard coordinate system into cartesian coordinates
    :param x: erhard coordinates
    :param L: tether length
    :return: cartesian coordinates
    """
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    if x.shape[1] == 4:
        L = x[:, 3]
    x_c = L * np.cos(x[:, 0])
    y_c = L * np.sin(x[:, 0]) * np.sin(x[:, 1])
    z_c = L * np.sin(x[:, 0]) * np.cos(x[:, 1])
    return np.array([x_c, y_c, z_c]).transpose()


def euler_step(f: Union[Callable, ca.MX], t, x, u, h: float):
    """
    evaluates a single euler integration step: xk+1 = xk + h * f(t, x, u)
    :param f: dgl right side
    :param t: time
    :param x: state
    :param u: input
    :param h: step size
    :return: new state
    """
    return x + h * f(t, x, u)


def rk4_step(f: Union[Callable, ca.MX], t, x, u, h: float):
    """
    evaluates a single euler integration step: xk+1 = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    :param f: dgl right side
    :param t: time
    :param x: state
    :param u: input
    :param h: step size
    :return: new state
    """
    k1 = f(t, x, u)
    k2 = f(t + h / 2, x + h / 2 * k1, u)
    k3 = f(t + h / 2, x + h / 2 * k2, u)
    k4 = f(t + h, x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class NLPConstraints(TypedDict):
    xlb: List
    xub: List
    xlb_terminal: List
    xub_terminal: List
    ulb: List
    uub: List
    stage_constrains: Optional[Callable[[ca.MX, ca.MX], ca.MX]]
    stage_constrains_bounds: Optional[Tuple[List, List]]
    terminal_constrains: Optional[Callable[[ca.MX], ca.MX]]
    terminal_constrains_bounds: Optional[Tuple[List, List]]


class ShootingNLP:
    def __init__(self, N: int, prediction_model: PredictionModel, stage_cost: Callable, terminal_cost: Callable,
                 delta_u_cost: Callable, constrains: NLPConstraints, use_soft_constraint: bool):
        self.N = N
        self.prediction_model = prediction_model
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.delta_u_cost = delta_u_cost
        self.stage_constrains = constrains['stage_constrains']
        self.stage_constrains_bounds = constrains['stage_constrains_bounds']
        self.terminal_constrains = constrains['terminal_constrains']
        self.terminal_constrains_bounds = constrains['terminal_constrains_bounds']
        self.uub = constrains['uub']
        self.ulb = constrains['ulb']

        self.xub = constrains['xub']
        self.xlb = constrains['xlb']
        self.xub_terminal = constrains['xub_terminal']
        self.xlb_terminal = constrains['xlb_terminal']
        self.use_soft_constraint = use_soft_constraint

    def get_cost(self, x_init: np.ndarray, u: np.ndarray):
        raise NotImplementedError("get cost of NLP is not implemented yet.")

    @abstractmethod
    def solve(self, t: float, x_init: np.ndarray, x0: np.ndarray, u0: np.ndarray,
              cost_options: Dict) -> Tuple[np.ndarray, np.ndarray, float, Dict, bool]:
        """ updates and solves the NLP

        :param t: initial time
        :param x_init: fixed initial value
        :param x0: initial guess for x
        :param u0: initial guess for u
        :param cost_options: additional names parameters for the cost function.
                             will be passed on to the calls of stage_ and terminal_cost:
                             stage_cost(Xk, Uk, k, **cost_options) and terminal_cost(XN, **cost_options)
        :return: x, u, loss, sol, success (state solution, input solution, total loss, casadi nlp solution,
                                           whether constrains are satisfied)
        """


class DirectMultipleShootingNLP(ShootingNLP):
    def __init__(self, N: int, prediction_model: PredictionModel, stage_cost: Callable, terminal_cost: Callable,
                 delta_u_cost: Callable, constrains: NLPConstraints, use_soft_constraint: bool):
        super().__init__(N, prediction_model, stage_cost, terminal_cost, delta_u_cost, constrains, use_soft_constraint)
        self.X = None
        self.U = None
        self.W = None
        self.Wlb = None
        self.Wub = None
        self.g = None
        self.glb = None
        self.gub = None

        self.slack_variables = None
        self.setup(0)

    def setup(self, t):
        U = []
        X = []
        slack_variables = []

        W = []
        Wlb = []
        Wub = []

        g = []
        glb = []
        gub = []

        X0 = ca.MX.sym('X0', len(self.xlb))
        X.append(X0)
        W.append(X0)
        Wlb.append(None)
        Wub.append(None)

        for i in range(self.N):
            Uk = ca.MX.sym(f'U_{i}', len(self.ulb))
            U.append(Uk)
            W.append(Uk)
            Wlb.append(self.ulb)
            Wub.append(self.uub)
            if i > 0:
                # additional constrains
                if self.stage_constrains is not None:
                    if self.use_soft_constraint:
                        slack_variable = ca.MX.sym(f'slack_{i}')
                        Wlb.append([0])
                        Wub.append([15])
                        W.append(slack_variable)
                        slack_variables.append(slack_variable)

                        g.append(self.stage_constrains(X[-1], U[-1]) + slack_variable)
                    else:
                        g.append(self.stage_constrains(X[-1], U[-1]))

                    glb.append(self.stage_constrains_bounds[0])
                    gub.append(self.stage_constrains_bounds[1])

            # model dynamics
            X_pred = self.prediction_model.predict(t + i * self.prediction_model.Ts, X[-1], U[-1])

            # new symbolic state
            X_new = ca.MX.sym(f'X_{i + 1}', len(self.xlb))
            X.append(X_new)
            W.append(X_new)
            if not i == self.N - 1:
                Wlb.append(self.xlb)
                Wub.append(self.xub)
            else:
                Wlb.append(self.xlb_terminal)
                Wub.append(self.xub_terminal)

            # dynamic constrain
            g.append(X_pred - X_new)
            glb.append([0] * len(self.xlb))
            gub.append([0] * len(self.xlb))

        # terminal constrains
        if self.terminal_constrains is not None:
            if self.use_soft_constraint:
                slack_variable = ca.MX.sym(f'slack_N')
                Wlb.append([0])
                Wub.append([15])
                W.append(slack_variable)
                slack_variables.append(slack_variable)

                g.append(self.terminal_constrains(X[-1]) + slack_variable)
            else:
                g.append(self.terminal_constrains(X[-1]))
            glb.append(self.terminal_constrains_bounds[0])
            gub.append(self.terminal_constrains_bounds[1])

        self.slack_variables = slack_variables if slack_variables else None
        self.X = X
        self.U = U
        self.W = W
        self.Wlb = Wlb
        self.Wub = Wub
        self.g = g
        self.glb = glb
        self.gub = gub

    def get_cost(self, x_init: np.ndarray, u: np.ndarray):
        raise NotImplementedError("get cost of NLP is not implemented yet.")

    def solve(self, t: float, x_init: np.ndarray, x0: np.ndarray, u0: np.ndarray, cost_options: Dict):
        # todo uncomment if time dependent
        # self.setup(t)
        W0 = []
        J = 0
        W0.append(x_init.tolist())
        self.Wlb[0] = x_init.tolist()
        self.Wub[0] = x_init.tolist()

        for i in range(self.N):
            W0.append(u0[i])

            # loss
            J += self.stage_cost(self.X[i], self.U[i], i, **cost_options)
            if self.slack_variables:
                J += 100 * (1 - ca.exp(-self.slack_variables[i] / 2))

            if self.slack_variables and i > 0:
                W0.append([0])
            W0.append(x0[i + 1])

        if self.slack_variables:
            W0.append([0])
        # terminal loss
        J += self.terminal_cost(self.X[-1], **cost_options)
        if self.slack_variables:
            J += 100 * (1 - ca.exp(-self.slack_variables[-1] / 2))

        # delta_u loss
        J += self.delta_u_cost(ca.horzcat(*self.U).T)

        nlp = {'x': ca.vertcat(*self.W), 'f': J, 'g': ca.vertcat(*self.g)}

        solver = ca.nlpsol("solver", "ipopt", nlp, {'ipopt.print_level': 0, 'print_time': 0, 'error_on_fail': False})

        sol = solver(lbx=np.concatenate(self.Wlb),
                     ubx=np.concatenate(self.Wub),
                     lbg=np.concatenate(self.glb),
                     ubg=np.concatenate(self.gub),
                     x0=np.concatenate(W0))

        constraints_satisfied = np.alltrue(np.concatenate(self.gub) + 1e-6 >= (sol['g'])) and np.alltrue(
            np.concatenate(self.glb) - 1e-6 <= (sol['g']))
        x = ca.Function('get_x', [nlp['x']], [ca.horzcat(*self.X).T])(sol['x']).full()
        u = ca.Function('get_x', [nlp['x']], [ca.horzcat(*self.U).T])(sol['x']).full()
        # slack = ca.Function('get_slack', [nlp['x']], [ca.horzcat(*self.slack_variables).T])(sol['x']).full()
        loss = ca.Function('get_x', [nlp['x']], [J])(sol['x']).full().item()

        # if not constrains_satisfied:
        #     return x, u0, loss, sol, constrains_satisfied

        return x, u, loss, sol, constraints_satisfied


class DirectSingleShootingNLP(ShootingNLP):
    def __init__(self, N: int, prediction_model: PredictionModel, stage_cost: Callable, terminal_cost: Callable,
                 delta_u_cost: Callable, constrains: NLPConstraints, use_soft_constraint: bool):
        super().__init__(N, prediction_model, stage_cost, terminal_cost, delta_u_cost, constrains, use_soft_constraint)

    def solve(self, t: float, x_init: np.ndarray, _, u0: np.ndarray, cost_options: Dict):
        if len(u0) != u0.shape[0]:
            raise RuntimeError('number of initial guesses dos not match with prediction horizon')
        U = []
        ulb = []
        uub = []
        g = []
        glb = []
        gub = []
        J = 0
        x = [x_init]
        slack_variables = []
        svub = []
        svlb = []

        for i in range(self.N):
            U.append(ca.MX.sym(f'U_{i}', u0.shape[1]))
            ulb.append(self.ulb)
            uub.append(self.uub)
            # loss
            J += self.stage_cost(x[-1], U[-1], i, **cost_options)

            if i > 0:
                # additional constrains
                if self.stage_constrains is not None:
                    if self.use_soft_constraint:
                        slack_variable = ca.MX.sym(f'slack_{i}')
                        svlb.append([0])
                        svub.append([15])
                        slack_variables.append(slack_variable)

                        g.append(self.stage_constrains(x[-1], U[-1]) + slack_variable)
                    else:
                        g.append(self.stage_constrains(x[-1], U[-1]))
                    glb.append(self.stage_constrains_bounds[0])
                    gub.append(self.stage_constrains_bounds[1])

                    # add slack cost
                    if self.use_soft_constraint:
                        J += 100 * (1 - ca.exp(-slack_variables[-1] / 2))

                # state constrains
                g.append(x[-1])
                glb.append(self.xlb)
                gub.append(self.xub)

            # model dynamics. Calculate x with respect to u
            x_new = self.prediction_model.predict(t + i * self.prediction_model.Ts, x[-1], U[-1])
            x.append(x_new)

        # terminal loss
        J += self.terminal_cost(x[-1], **cost_options)

        # delta_u loss
        J += self.delta_u_cost(ca.horzcat(*U).T)

        # terminal constrains
        if self.terminal_constrains is not None:
            if self.use_soft_constraint:
                slack_variable = ca.MX.sym(f'slack_N')
                svlb.append([0])
                svub.append([15])
                slack_variables.append(slack_variable)

                g.append(self.terminal_constrains(x[-1]) + slack_variable)
            else:
                g.append(self.terminal_constrains(x[-1]))

            glb.append(self.terminal_constrains_bounds[0])
            gub.append(self.terminal_constrains_bounds[1])

            if self.use_soft_constraint:
                J += 100 * (1 - ca.exp(-slack_variables[-1] / 2))

        g.append(x[-1])
        glb.append(self.xlb_terminal)
        gub.append(self.xub_terminal)

        X = ca.horzcat(*x).T
        if slack_variables:
            x_nlp = ca.vertcat(*(U + slack_variables))
            x0_nlp = np.concatenate((u0.flatten(), np.zeros(len(slack_variables))))
            ubx = np.concatenate(uub + svub)
            lbx = np.concatenate(ulb + svlb)
        else:
            x_nlp = ca.vertcat(*U)
            x0_nlp = u0.flatten()
            ubx = np.concatenate(uub)
            lbx = np.concatenate(ulb)

        nlp = {'x': x_nlp, 'f': J, 'g': ca.vertcat(*g)}

        solver = ca.nlpsol("solver", "ipopt", nlp, {'ipopt.print_level': 0, 'print_time': 0})

        sol = solver(lbx=lbx,
                     ubx=ubx,
                     lbg=np.concatenate(glb),
                     ubg=np.concatenate(gub),
                     x0=x0_nlp)
        constrains_satisfied = np.alltrue(np.concatenate(gub) + 1e-6 >= (sol['g'])) and np.alltrue(
            np.concatenate(glb) - 1e-6 <= (sol['g']))

        x = ca.Function('get_x', [nlp['x']], [X])(sol['x']).full()
        u = ca.Function('get_u', [nlp['x']], [ca.horzcat(*U).T])(sol['x']).full()
        loss = ca.Function('get_loss', [nlp['x']], [J])(sol['x']).full().item()

        return x, u, loss, sol, constrains_satisfied


def iterate_prediction_model(t: float, x0: np.ndarray, N: int, prediction_model: PredictionModel, u_input: np.ndarray):
    """
    generates the state trajectory for a given input and prediction model
    :param t: initial time
    :param x0: initial state
    :param N: number of prediction steps
    :param prediction_model: model used to do the prediction
    :param u_input: input trajectory
    :return: state trajectory
    """
    x = np.zeros((N + 1, len(x0)))
    x[0] = x0
    for i in range(1, N + 1):
        x[i] = prediction_model.predict(t + (i - 1) * prediction_model.Ts, x[i - 1], u_input[i - 1])
    return x


class ControlPath:
    def __init__(self, trajectory):

        x = trajectory.x

        self.trajectory = trajectory
        self.path_speed = 1 / self.trajectory.t[-1]
        self.n = x.shape[1]

        self.splines_np = [CubicSpline(np.linspace(0, 1, len(x)), x[:, i], bc_type='periodic') for i in
                           range(self.n)]
        self.splines = [ca.interpolant('LUT', 'bspline', [np.linspace(0, 1, len(x))], x[:, i]) for i in
                        range(self.n)]

    def __call__(self, *args, **kwargs):
        s = args[0]
        angles = []
        for i in range(self.n):
            if isinstance(s, float) or isinstance(s, int):
                angles.append(self.splines[i](s % 1).full().item())
            elif isinstance(s, np.ndarray):
                angles.append(self.splines[i](s % 1).full())
            else:
                angles.append(self.splines[i](ca.mod(s, 1)))
        if isinstance(s, float) or isinstance(s, int):
            return np.array(angles)
        elif isinstance(s, np.ndarray):
            return np.hstack(angles)
        else:
            return ca.vertcat(*angles)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
