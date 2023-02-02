""" This module defines the models used in this Thesis.

Note that there are different prediction_model getter function which return different prediction_model objects with the
same dynamics. That is necessary because casadi needs its own functions and won't work with the numpy implementation.

"""
import numpy as np
from numpy import cos, sin, sqrt, tan
import casadi as ca
from typing import Callable, TypedDict, List, Tuple, Union
import utils
from scipy.linalg import expm
from scipy.integrate import quad_vec
from interfaces import ResidualModel
from NN_modeling import pytorch_NN


class ErhardParameter(TypedDict):
    A: float
    L: float
    c_tilde: float
    v0: float
    E0: float
    beta: float
    rho: float


class TowingKiteConstraints(TypedDict):
    u_min: List[float]
    u_max: List[float]
    h_min: float
    x_lb: List[float]
    x_ub: List[float]


class ErhardCasadiModel(TypedDict):
    t: ca.MX
    x: ca.MX
    u: ca.MX
    f: ca.Function
    height_function: Callable
    TF: ca.MX
    parameter: ErhardParameter
    xdot: ca.MX


# model parameter
ERHARD_PHYSICAL_PARAMETER: ErhardParameter = {'A': 300, 'L': 400, 'c_tilde': 0.028, 'v0': 10, 'E0': 5, 'beta': 0,
                                              'rho': 1}
ERHARD_PLANT_PARAMETER: ErhardParameter = {'A': 300, 'L': 400, 'c_tilde': 0.06, 'v0': 15, 'E0': 5.5, 'beta': 0,
                                           'rho': 1}
ERHARD_PLANT_PARAMETER_2: ErhardParameter = {'A': 300, 'L': 400, 'c_tilde': 0.06, 'v0': 15, 'E0': 8, 'beta': 0,
                                             'rho': 1}
ERHARD_PLANT_PARAMETER_3: ErhardParameter = {'A': 293, 'L': 408, 'c_tilde': 0.07, 'v0': 19, 'E0': 7, 'beta': 0.2,
                                             'rho': 1.02}

PUMPING_PROTOTYPE_PARAMETER = {'A': 21.0, 'CR': 1.0, 'v0': 10, 'E': 5, 'rho': 1.2, 'gk': 0.1}

ERHARD_Ts: float = 0.27
ERHARD_STATE_ORDER: List[str] = ['theta', 'phi', 'psi']
ERHARD_NOISE_STD_DEG_LOW = np.array([0.64, 0.99, 2.4]) * 1 / 3 * 0.002 * 180 / np.pi
ERHARD_NOISE_STD_DEG_MEDIUM = ERHARD_NOISE_STD_DEG_LOW * 5
ERHARD_NOISE_STD_DEG_HIGH = ERHARD_NOISE_STD_DEG_MEDIUM * 5
ERHARD_NOISE_STD_DEG_SUPER_HIGH = ERHARD_NOISE_STD_DEG_HIGH * 3

ERHARD_NOISE_THRUST_LOW = 500
ERHARD_NOISE_THRUST_MEDIUM = ERHARD_NOISE_THRUST_LOW * 5
ERHARD_NOISE_THRUST_HIGH = ERHARD_NOISE_THRUST_MEDIUM * 5
ERHARD_NOISE_THRUST_SUPER_HIGH = ERHARD_NOISE_THRUST_HIGH * 3

# input and state constraints

KITE_CONSTRAINTS: TowingKiteConstraints = {'u_min': [-10], 'u_max': [10], 'h_min': 100,
                                           'x_lb': [0, -np.pi / 2, -np.pi],
                                           'x_ub': [np.pi / 2, np.pi / 2, np.pi]}

PUMPING_PROTOTYPE_CONSTRAINS = {'delta_max': 0.7, 'delta_dot_max': 0.7, 'l_max': 300, 'theta_min': 0.35, 'va_min': 5.0,
                                'v_winch_min': -5.0}


def __unpack_parameter(parameter_set):
    L = parameter_set['L']
    A = parameter_set['A']
    c_tilde = parameter_set['c_tilde']
    v0 = parameter_set['v0']
    E0 = parameter_set['E0']
    beta = parameter_set['beta']
    rho = parameter_set['rho']
    PD = rho * v0 ** 2 / 2
    return L, A, c_tilde, v0, E0, beta, PD


def __right_side(va, L, theta, psi, E, u):
    theta_dot = va / L * (cos(psi) - (tan(theta)) / E)
    phi_dot = -va / (L * sin(theta)) * sin(psi)
    psi_dot = va / L * u + phi_dot * cos(theta)
    return theta_dot, phi_dot, psi_dot


def __get_va_and_E(E0, c_tilde, u, v0, theta):
    E = E0 - c_tilde * u ** 2
    va = v0 * E * cos(theta)
    return va, E


def __get_thrust(PD, A, E, beta, theta, phi):
    TF = (PD * A * (cos(theta)) ** 2 * (E + 1) * sqrt(E ** 2 + 1)) * (
            cos(theta) * cos(beta) + sin(theta) * sin(beta) * sin(phi))
    return TF


def __get_height(L, theta, phi):
    height = L * sin(theta) * cos(phi)
    return height


def get_pumping_quaternions_casadi_model(parameter_set):
    A, CR, v0, E, rho, gk = tuple([parameter_set[name] for name in ['A', 'CR', 'v0', 'E', 'rho', 'gk']])
    gamma_q = 0.01

    # Model Variables
    q = ca.MX.sym('q', (4, 1))
    l = ca.MX.sym('l')
    delta = ca.MX.sym('delta')
    delta_dot = ca.MX.sym('delta_dot')
    v_winch = ca.MX.sym('v_winch')
    W = ca.MX.sym('W')

    va = v0 * E * (q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2) - v_winch * E

    q1_dot = va / (2 * l) * (-q[2]) + v0 / l * q[0] * (q[2] ** 2 + q[3] ** 2) + gk * va * delta / 2 * q[1]
    q2_dot = va / (2 * l) * (-q[3]) + v0 / l * q[1] * (q[2] ** 2 + q[3] ** 2) + gk * va * delta / 2 * -q[0]
    q3_dot = va / (2 * l) * (q[0]) + v0 / l * q[2] * (-q[0] ** 2 + q[1] ** 2) + gk * va * delta / 2 * -q[3]
    q4_dot = va / (2 * l) * (q[1]) + v0 / l * q[3] * (-q[0] ** 2 + q[1] ** 2) + gk * va * delta / 2 * q[2]
    l_dot = v_winch
    q_dot = ca.vertcat(q1_dot, q2_dot, q3_dot, q4_dot) - gamma_q * (ca.dot(q, q) - 1) * q
    W_dot = l_dot * va ** 2

    x = ca.vertcat(W, delta, l, q)
    x_dot = ca.vertcat(W_dot, delta_dot, l_dot, q_dot)
    u = ca.vertcat(delta_dot, v_winch)

    # Continuous Time Dynamics
    f = ca.Function('f', [x, u], [x_dot])

    return {'x': x, 'u': u, 'f': f, 'va_function': lambda x, u: v0 * E * cos(x[0]) - u[1] * E,
            'parameter': parameter_set, 'xdot': x_dot}


def get_pumping_casadi_model(parameter_set):
    A, CR, v0, E, rho, gk = tuple([parameter_set[name] for name in ['A', 'CR', 'v0', 'E', 'rho', 'gk']])

    # Model Variables
    x = ca.MX.sym('x', (4, 1))
    u = ca.MX.sym('u', (2, 1))
    t = ca.MX.sym('t')

    v_winch = u[1]
    va = v0 * E * cos(x[0]) - v_winch * E

    theta_dot, phi_dot, psi_dot = __right_side(va, x[3], x[0], x[2], E, u[0] * gk)
    xdot = ca.vertcat(theta_dot, phi_dot, psi_dot, v_winch)

    F_tether = rho * A * CR / 2 * (1 + E ** 2) / E ** 2 * va ** 2

    # Continuous Time Dynamics
    f = ca.Function('f', [t, x, u], [xdot])

    return {'t': t, 'x': x, 'u': u, 'f': f, 'height_function': lambda state: __get_height(state[3], state[0], state[1]),
            'TF': F_tether, 'parameter': ERHARD_PHYSICAL_PARAMETER, 'xdot': xdot}


def get_erhard_casadi_model(parameter_set: ErhardParameter,
                            wind_function: Callable[[float], float] = None) -> ErhardCasadiModel:
    """ returns the erhard model with the given parameters in a form that is suited to work with casadi.

    :param parameter_set: contains all model parameter
    :param wind_function: see model_zoo.get_erhard_dgl_and_thrust_function
    :return: corresponding casadi model
    """
    L, A, c_tilde, v0, E0, beta, PD = __unpack_parameter(parameter_set)

    # Model Variables
    x = ca.MX.sym('x', (3, 1))
    u = ca.MX.sym('u')
    t = ca.MX.sym('t')

    # Dynamic Model Equations
    if wind_function:
        va, E = __get_va_and_E(E0, c_tilde, u, wind_function(t) * v0, x[0])
    else:
        va, E = __get_va_and_E(E0, c_tilde, u, v0, x[0])

    theta_dot, phi_dot, psi_dot = __right_side(va, L, x[0], x[2], E, u)
    xdot = ca.vertcat(theta_dot, phi_dot, psi_dot)
    TF: ca.MX = __get_thrust(PD, A, E, beta, x[0], x[1])

    # Continuous Time Dynamics
    f = ca.Function('f', [t, x, u], [xdot])

    return {'t': t, 'x': x, 'u': u, 'f': f, 'height_function': lambda state: __get_height(L, state[0], state[1]),
            'TF': TF, 'parameter': parameter_set, 'xdot': xdot}


def get_erhard_dgl_and_thrust_function(
        parameter_set: ErhardParameter,
        wind_function: Callable[[float], float] = None,
        drift_function: Callable[[float], float] = None) -> Tuple[Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                                                                  Callable[[float, np.ndarray, np.ndarray], float]]:
    """ returns the dynamic and thrust function for a given parameter set

    :param parameter_set: containing all parameters
    :param wind_function: gives v0(t)/parameter_set['v0']
    :param drift_function: gives E0(t)/parameter_set['E0']
    :return: right side of dgl and thrust function
    """
    L, A, c_tilde, v0, E0_const, beta, _ = __unpack_parameter(parameter_set)
    if wind_function is None:
        wind_function = lambda t: 1
    if drift_function is None:
        drift_function = lambda t: 1
    rho = parameter_set['rho']

    def f(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta = x[0]
        psi = x[2]
        E0 = drift_function(t) * E0_const

        # Dynamic Model Equations
        va, E = __get_va_and_E(E0, c_tilde, u.item(), wind_function(t) * v0, theta)
        theta_dot, phi_dot, psi_dot = __right_side(va, L, theta, psi, E, u.item())

        return np.array([theta_dot, phi_dot, psi_dot])

    def thrust(t: float, x: np.ndarray, u: np.ndarray) -> float:
        theta = x[0]
        phi = x[1]
        E0 = drift_function(t) * E0_const

        va, E = __get_va_and_E(E0, c_tilde, u.item(), wind_function(t) * v0, theta)
        PD = rho * (wind_function(t) * v0) ** 2 / 2
        TF = __get_thrust(PD, A, E, beta, theta, phi)

        return TF

    return f, thrust


def get_linear_erhard_model(parameter_set: ErhardParameter,
                            x_op: np.ndarray,
                            u_op: float) -> Tuple[np.ndarray, np.ndarray]:
    """ linearizes the erhard model for a given parameter set and a given operation point.

    :param parameter_set: contains the model parameter
    :param x_op: operation state
    :param u_op: operation input
    :return: A, B (Tuple of system and input matrix)
    """
    # # ______ verify linearisation _______
    # x_op = np.array([0.6, 0.8, 1.2])
    # u_op = 4
    # parameter_set = ERHARD_PHYSICAL_PARAMETER
    # f, _ = get_erhard_dgl_and_thrust_function(parameter_set)
    # h = 0.000001
    # dx1 = np.array([h, 0, 0])
    # dx2 = np.array([0, h, 0])
    # dx3 = np.array([0, 0, h])
    # df_dx1 = (f(-1, x_op + dx1, u_op) - f(-1, x_op, u_op)) / h
    # df_dx2 = (f(-1, x_op + dx2, u_op) - f(-1, x_op, u_op)) / h
    # df_dx3 = (f(-1, x_op + dx3, u_op) - f(-1, x_op, u_op)) / h
    # A_num = np.concatenate((df_dx1, df_dx2, df_dx3)).reshape(3, 3)
    #
    # B_num = (f(-1, x_op, u_op + h) - f(-1, x_op, u_op)) / h

    L, A, c_tilde, v0, E0, beta, PD = __unpack_parameter(parameter_set)
    # get dynamics in operation point
    va_op, E_op = __get_va_and_E(E0, c_tilde, u_op, v0, x_op[0])
    dva_dtheta = - v0 * E_op * sin(x_op[0])
    a11 = dva_dtheta / L * (cos(x_op[2]) - tan(x_op[0]) / E_op) - va_op / (L * E_op) * 1 / cos(x_op[0]) ** 2
    a12 = -dva_dtheta / (L * sin(x_op[0])) * sin(x_op[2]) - va_op / L * sin(x_op[2]) * -1 / sin(x_op[0]) ** 2 * cos(
        x_op[0])
    a13 = dva_dtheta / L * u_op + a12 * cos(x_op[0]) + -va_op / (L * sin(x_op[0])) * sin(x_op[2]) * -sin(x_op[0])

    a21 = 0
    a22 = 0
    a23 = 0

    a31 = va_op / L * -sin(x_op[2])
    a32 = -va_op / (L * sin(x_op[0])) * cos(x_op[2])
    a33 = a32 * cos(x_op[0])

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    b1 = -v0 / L * c_tilde * 2 * u_op * cos(x_op[0]) * cos(x_op[2])
    b2 = c_tilde * 2 * u_op * v0 * cos(x_op[0]) / (sin(x_op[0]) * L) * sin(x_op[2])
    b3 = (v0 * (c_tilde * -3 * u_op ** 2 + E0) / L + b2) * cos(x_op[0])

    B = np.array([b1, b2, b3])
    return A, B


class PredictionModel:
    def __init__(self,
                 dgl_right_side: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 Ts: float,
                 casadi_model: ErhardCasadiModel = None,
                 residual_state_model: ResidualModel = None,
                 thrust_function: Callable[[float, np.ndarray, np.ndarray], float] = None,
                 residual_value_model: ResidualModel = None,
                 steps: int = 1,
                 discretization_method: str = 'rk4'):
        """ Prediction model used in the MPC Controller.
        This can be used to make predictions for numeric values (numpy arrays) as well as symbolic (ca.sym.MX)

        :param dgl_right_side: right side of the dgl
        :param Ts: sampling time (here equal to prediction time)
        :param casadi_model: casadi model
        :param residual_state_model: model to estimate the model plant missmatch
        :param thrust_function: thrust function depending on the state and input.
        :param residual_value_model: model to estimate the error in the integrated thrust over one time step
        :param steps: the number prediction steps used per sampling time (can be used to reduce numerical errors)
        :param discretization_method: method used to solve the dgl (euler or rk4)
        """
        self.dgl_right_side = dgl_right_side
        self.thrust_function = thrust_function
        self.casadi_model = casadi_model
        self.residual_state_model = residual_state_model
        self.residual_value_model = residual_value_model
        self.Ts = Ts
        self.steps = steps
        self.h = self.Ts / steps

        self.discretization_method = discretization_method
        if discretization_method == 'euler':
            self.step_function = utils.euler_step
        elif self.discretization_method == 'rk4':
            self.step_function = utils.rk4_step
        self.function = ca.Function('thrust', [casadi_model['t'], casadi_model['x'], casadi_model['u']],
                                    [casadi_model['TF'] * Ts])

    def predict(self, t: Union[float, ca.MX], x: Union[np.ndarray, ca.MX], u: Union[np.ndarray, ca.MX],
                ignore_residual_model: bool = False) -> Union[np.ndarray, ca.MX]:
        """ predicts the next state.

        :param t: tk
        :param x: xk
        :param u: uk
        :param ignore_residual_model: ignores the correction of the residual model.
        :return: xk+1
        """
        if isinstance(x, ca.MX) or isinstance(u, ca.MX):
            f = self.casadi_model['f']
        else:
            f = self.dgl_right_side

        x_next = x
        for i in range(self.steps):
            x_next = self.step_function(f, t, x_next, u, self.h)
        if self.residual_state_model is not None and not ignore_residual_model:
            x_next += self.residual_state_model.predict(x, u)
        return x_next

    def predict_thrust(self, t: Union[float, ca.MX], x: Union[np.ndarray, ca.MX],
                       u: Union[np.ndarray, ca.MX]) -> Union[np.ndarray, ca.MX]:
        """
        predicts the integral of the thrust over one time step.
        :param t: tk
        :param x: xk
        :param u: uk
        :return: integral_tk^tk+1(trust(x, uk))
        """
        if isinstance(x, ca.MX) or isinstance(u, ca.MX):
            thrust = self.function(t, x, u)
        else:
            thrust = self.thrust_function(t, x, u) * self.Ts

        if self.residual_value_model is not None:
            thrust += self.residual_value_model.predict(x, u)
        return thrust


class VirtualSystem:

    # noinspection PyTupleAssignmentBalance
    def __init__(self, n_states: int, Ts: float):
        """
        Simple discrete integrator VirtualSystem for path following MPC.
        :param n_states: number of states
        :param Ts: sampling time
        """
        self.state = ca.MX.sym('state', n_states)
        self.Ts = Ts
        self.virtual_input = ca.MX.sym('virtual_input')
        self.A = np.array(
            [[0 if j + 1 != k else 1 for k in range(n_states)] for j in range(n_states)])  # integrator chain
        self.b = np.array([0 if k < n_states - 1 else 1 for k in range(n_states)])

        self.state_dot = self.A @ self.state + self.b * self.virtual_input

        self.A_d = expm(self.A * Ts)
        self.b_d, _ = quad_vec(lambda t: expm(self.A * t) @ self.b, 0, Ts)

    def predict(self, state: Union[np.ndarray, ca.MX], virtual_input: Union[float, ca.MX]):
        """
        predicts the next virtual state
        :param state: xk_virtual
        :param virtual_input: uk_virtual
        :return: xk+1_virtual
        """
        return self.A_d @ state + self.b_d * virtual_input


class ExtendedPredictionModel(PredictionModel):
    def __init__(self, dgl_right_side: Callable[[float, np.ndarray, np.ndarray], np.ndarray], Ts: float,
                 casadi_model: ErhardCasadiModel = None,
                 residual_state_model: ResidualModel = None,
                 thrust_function: Callable[[float, np.ndarray, np.ndarray], float] = None,
                 residual_value_model: ResidualModel = None, steps: int = 10,
                 discretization_method: str = 'rk4', n_virtual_states: int = 3):
        """
        Combines a prediction model with a virtual system.

        :param dgl_right_side: see PredictionModel
        :param Ts: see PredictionModel, Virtual system
        :param casadi_model: see PredictionModel
        :param residual_state_model: see PredictionModel
        :param thrust_function: see PredictionModel
        :param residual_value_model: see PredictionModel
        :param steps: see PredictionModel
        :param discretization_method: see PredictionModel
        :param n_virtual_states: see Virtual system
        """
        super().__init__(dgl_right_side,
                         Ts,
                         casadi_model,
                         residual_state_model,
                         thrust_function,
                         residual_value_model,
                         steps,
                         discretization_method)

        self.virtual_system = VirtualSystem(n_virtual_states, Ts)

        self.z = ca.vertcat(self.casadi_model['x'], self.virtual_system.state)
        self.v = ca.vertcat(self.casadi_model['u'], self.virtual_system.virtual_input)

        self.nx = self.casadi_model['x'].shape[0]
        self.mx = self.casadi_model['u'].shape[0]
        self.nx_virtual = self.virtual_system.state.shape[0]
        self.mx_virtual = self.virtual_system.virtual_input.shape[0]

    def predict(self, t: Union[float, ca.MX], z: Union[np.ndarray, ca.MX], v: Union[float, np.ndarray, ca.MX],
                ignore_residual_model: bool = False, ignore_virtual_system: bool = False) -> Union[np.ndarray, ca.MX]:
        """
        predicts the next state of the extended system
        :param t: current time
        :param z: extended state
        :param v: extended input
        :param ignore_residual_model: ignores the residual model
        :param ignore_virtual_system: ignores the virtual system and calls predict of the not extended prediction model
        :return: zk+1 or xk+1 if ignore_virtual_system
        """
        if ignore_virtual_system:
            return super().predict(t, z, v, ignore_residual_model=ignore_residual_model)
        x_pred = super().predict(t, z[:self.nx], v[:self.mx], ignore_residual_model=ignore_residual_model)
        if isinstance(z, ca.MX) or isinstance(v, ca.MX):
            virtual_x_pred = self.virtual_system.predict(z[self.nx:], v[self.mx:])
            return ca.vertcat(x_pred, virtual_x_pred)
        else:
            virtual_x_pred = self.virtual_system.predict(z[self.nx:].flatten(), v[self.mx:].item())
            return np.concatenate((x_pred, virtual_x_pred))


def get_Erhard_prediction_model(system_name: str = 'physical',
                                steps: int = 1, discretization_method: str = 'rk4',
                                n_virtual_states: int = None, time_variant: bool = False,
                                wind_function: Callable[[float], float] = None, state_GP_name: Union[str, None] = '',
                                thrust_GP_name: str = None) -> PredictionModel:
    """
    load the Erhard prediction model for the desired model and parameters.
    :param system_name: model name (physical or plant or hybrid)
    :param steps: see PredictionModel
    :param discretization_method: see PredictionModel
    :param n_virtual_states: number of virtual states if not None an ExtendedPredictionModel will be returned
    :param time_variant: whether the prediction_model is time variant
    :param wind_function: see model_zoo.get_erhard_dgl_and_thrust_function
    :param state_GP_name: name of the state gp
    :param thrust_GP_name: name of the thrust gp
    :return: PredictionModel object with the desired parameter and options.
    """
    if system_name in ['physical', 'hybrid']:
        parameter_set = ERHARD_PHYSICAL_PARAMETER
    elif system_name == 'plant':
        parameter_set = ERHARD_PLANT_PARAMETER
    else:
        raise RuntimeError(f'unknown system name: {system_name}')

    if time_variant:
        f, trust = get_erhard_dgl_and_thrust_function(parameter_set, wind_function=wind_function)
        casadi_model = get_erhard_casadi_model(parameter_set, wind_function)
    else:
        f, trust = get_erhard_dgl_and_thrust_function(parameter_set)
        casadi_model = get_erhard_casadi_model(parameter_set)

    Ts = ERHARD_Ts

    state_GP = load_residual_model(state_GP_name) if system_name == 'hybrid' and state_GP_name else None
    thrust_GP = load_residual_model(thrust_GP_name) if system_name == 'hybrid' and thrust_GP_name else None
    if n_virtual_states is None:
        prediction_model = PredictionModel(f, Ts, casadi_model, state_GP, trust, thrust_GP, steps,
                                           discretization_method)
    else:
        prediction_model = ExtendedPredictionModel(f, Ts, casadi_model, state_GP, trust, thrust_GP, steps,
                                                   discretization_method, n_virtual_states)
    return prediction_model


def load_residual_model(name: str, load_mlp=True) -> ResidualModel:
    return pytorch_NN.load_model(name)
