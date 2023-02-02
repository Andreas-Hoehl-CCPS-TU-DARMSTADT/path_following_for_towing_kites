"""Simulation framework.

This module provides a Simulator, which represents the interaction with the physical system.
One can implement controller and run experiments
"""
from scipy.integrate import solve_ivp
import numpy as np
from controller.basic_contoller import OpenLoop, Controller
from typing import Callable, Tuple
import time
from interfaces import SimulationResult
from tqdm import trange


class Simulator:

    def __init__(self, dgl: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                 Ts: float, u_min: np.ndarray, u_max: np.ndarray, controller: Controller = None,
                 value_function: Callable[[float, np.ndarray, np.ndarray], float] = None,
                 noise_function: Callable[[], np.ndarray] = None) -> None:
        """ creates Simulator object.

        :param dgl: right side of ode x_dot = dgl(t, x, u)
        :param Ts: sampling time
        :param u_min: minimal control output
        :param u_max: maximal control output
        :param controller: controller for closed loop, OpenLoop is used if controller is None
        :param value_function: Function to calculate a metric (eg. thrust)
        :param noise_function: Function to calculate the measurement noise
        """
        self.u_sim, self.t_sim, self.x_sim, self.x_measurements, self.u_controller = None, None, None, None, None
        self.k_index, self.internal_information = None, None
        self.Ts = Ts
        self.dgl = dgl
        self.value_function = value_function
        self.noise_function = noise_function
        self.u_min = u_min
        self.u_max = u_max

        if controller is None:
            self.controller = OpenLoop()
        else:
            self.controller = controller

        self.solve_method = 'RK45'

    def reset_simulation(self, x0: np.ndarray):
        """ resets the simulation and the controller.

        :param x0: initial state
        """
        self.u_sim = []
        self.u_controller = []
        self.t_sim = [0]
        self.x_sim = [x0]
        self.x_measurements = [x0]
        self.k_index = [0]
        self.internal_information = []

        self.controller.reset()

    def saturation(self, u: np.ndarray) -> np.ndarray:
        """ saturation od the control output.
        :param u: control output
        :return: saturated output
        """
        return np.clip(u, self.u_min, self.u_max)

    def simulate_time_step(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ simulates a single time step.

        :param t: time of the start of the time step (only relevant for time invariant systems)
        :param x: state before the time step
        :param u: control output
        :return: time vector (step size of solver), state vector during the time step (row wise)
        """
        u_sat = self.saturation(u)
        result = solve_ivp(self.dgl, [t, t + self.Ts], x, method=self.solve_method, args=(u_sat,), max_step=0.01)
        # max step size of 0.0001 gives slightly different results but takes significantly longer. But still feasible.

        y = result.y.transpose()

        return result.t, y

    def evaluate_value_function(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """ calculates the value function for a precalculated Trajectory.

        The calculation is done using the trapeze rule i.e. first order approximation
        :return: integrated value, integrated value for each time step, value function evaluated at self.t_sim
        """
        values = np.array(
            [self.value_function(self.t_sim[i], self.x_sim[i], self.u_sim[i]) for i in
             range(len(self.u_sim))])  # not very fast code
        # value = integrate.simpson(values, self.t_sim[:-1])
        # simpson rule does not work?! gives completely wrong values for uneven number of samples
        # ok this makes sense since the samples are not equidistant!

        # value = integrate.trapz(values, self.t_sim[:-1]) # equivalent to the code below

        t = np.array(self.t_sim)
        delta_t = t[1:-1] - t[:-2]
        mid_value = (values[:-1] + values[1:]) / 2
        time_step_value = np.array(
            [(delta_t[self.k_index[i]:self.k_index[i + 1]] * mid_value[self.k_index[i]:self.k_index[i + 1]]).sum() for i
             in range(len(self.k_index) - 1)])
        value = time_step_value.sum()  # =(delta_t * mid_value).sum()
        return value, time_step_value, values

    def run_simulation(self, T: float, x0: np.ndarray, delay_input: bool = False,
                       constant_u: np.ndarray = None, verbose: int = 0) -> SimulationResult:
        """ runs a simulation from t=0 to t=T.

        Fist resets the simulation and the controller. The controller is also reset at the end.
        :param T: simulation time
        :param x0: initial state
        :param delay_input: delays the input by one time step to account for calculation time if true.
                            First control output is then obtained using controller.get_init_u()
        :param constant_u: ignore the controller and simulate with the given value for u
        :param verbose: 0 -> minimal print out, 1 -> print current time
        :return: dict containing the result of the simulation see also SimulationResult
                 k_index
                    contains the list of the indices of the time steps
                 info
                    contains information of the controller after each time step collected via .collect_info
                 integrate_value
                    contains the integral of the value function
                 integrate_time_step_value
                    contains the integral of the value function for each time step
                 moment_values
                    contains the momentary value function values
                 measurements
                    contains the measured states (which are passed to the controller
                    and which contain noise calculated by the given noise function)
                u_controller
                    contains the control outputs
                computation_times
                    contains the time needed to compute the control output for each time step
        """
        self.reset_simulation(x0.copy())

        computation_times = []
        if verbose > 0:
            pbar = trange(int(T / self.Ts), unit='time step')
        else:
            pbar = range(int(T / self.Ts))
        for i in pbar:
            if verbose > 0:
                pbar.set_postfix(current_time=f'{i * self.Ts:.2f}/ {T}')
            if constant_u is not None:
                u_new = constant_u
            else:
                if delay_input:
                    if i == 0:
                        s = time.time()
                        u_new = self.controller.get_init_u()
                        computation_times.append(time.time() - s)
                    else:
                        s = time.time()
                        u_new = self.controller.calculate(self.x_measurements[-2].copy(), True)
                        computation_times.append(time.time() - s)
                else:
                    s = time.time()
                    u_new = self.controller.calculate(self.x_measurements[-1].copy(), False)
                    computation_times.append(time.time() - s)

            self.internal_information.append(self.controller.collect_info())

            t, x = self.simulate_time_step(self.t_sim[-1], self.x_sim[-1], u_new)

            # concatenation is slow!
            self.t_sim.extend(t[1:])  # first element is already in the list
            self.x_sim.extend(x[1:])

            # add noise
            if self.noise_function is None:
                noise = 0
            else:
                noise = self.noise_function()

            self.x_measurements.append(x[-1] + noise)
            self.u_controller.append(u_new)  # controller output before saturation
            self.u_sim.extend([self.saturation(u_new)] * (len(t) - 1))  # output after saturation
            self.k_index.append(self.k_index[-1] + len(t) - 1)  # can be used to get the vales for the given time steps

        if verbose > 0:
            pbar.close()
        self.controller.reset()

        if self.value_function is None:
            value = 0
            values = np.zeros(len(self.u_sim))
            time_step_value = np.zeros(len(self.u_controller))
        else:
            value, time_step_value, values = self.evaluate_value_function()

        result = {'x': np.array(self.x_sim), 'u': np.array(self.u_sim), 'u_controller': np.array(self.u_controller),
                  't': np.array(self.t_sim), 'k_index': np.array(self.k_index), 'info': self.internal_information,
                  'integrate_value': value, 'moment_values': values, 'integrate_time_step_value': time_step_value,
                  'measurements': np.array(self.x_measurements), 'computation_times': np.array(computation_times)}
        return result
