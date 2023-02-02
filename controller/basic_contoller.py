import warnings
from abc import abstractmethod
from typing import Dict
import numpy as np


class Controller:
    """ abstract class of a controller.

    """

    @abstractmethod
    def reset(self):
        """ resets the controller.

        """

    @abstractmethod
    def collect_info(self) -> Dict:
        """ method to collect internal information of a controller during the simulation.

        This method is called at the beginning and after each time step of a simulation and the return value is stored
        in a list.

        Returns
        -------
        dict
            dictionary that may contain internal information of the controller such as the state predictions.
        """

    @abstractmethod
    def calculate(self, x: np.ndarray, delay_u: bool) -> float:
        """calculates the control output for a given state_k.


        Parameters
        ----------
        x : np.ndarray
            current state_k (state_k-1 if delay_u)
        delay_u : bool
            whether the input should be delayed to make room for computations need in reality

        Returns
        -------
        float
            control output u_k

        """

    @abstractmethod
    def get_init_u(self) -> float:
        """returns the control output before any measurement is taken. Only used if input is delayed.

        Returns
        -------
        float
            control output u_0

        """

    @abstractmethod
    def get_x_init(self) -> np.ndarray:
        """returns the initial state of the controller

        Returns
        -------
        np.ndarray
            initial state x0

        """

    @abstractmethod
    def set_x_init(self, x0: np.ndarray):
        """sets the initial state of the controller

        Parameters
        ----------
        x0 : np.ndarray
            new initial state

        Returns
        -------

        """


class UForwardController(Controller):

    def __init__(self, u_opt: np.ndarray, x_init: np.ndarray) -> None:
        """feed forward controller that periodically repeats a reference trajectory.

        Parameters
        ----------
        u_opt : np.ndarray
            precalculated input sequence
        x_init : np.ndarray
            initial state of the controller
        """
        self.u_opt = u_opt
        self.x_init = x_init
        self.idx = 0

    def calculate(self, x, delay_u):
        u = self.u_opt[self.idx]
        if self.idx < len(self.u_opt) - 1:
            self.idx += 1
        else:
            self.idx = 0
        return u

    def collect_info(self):
        return {'idx': self.idx}

    def reset(self):
        self.idx = 0

    def get_init_u(self):
        if self.idx != 0:
            warnings.warn('method get_init_u() was called with idx != 0 !')
        self.idx += 1
        return self.u_opt[0]

    def get_x_init(self):
        return self.x_init

    def set_x_init(self, x0: np.ndarray):
        self.x_init = x0


class OpenLoop(Controller):
    """ Open loop controller i.e. the control output is always zero.

    """

    def __init__(self):
        self.x_init = None

    def reset(self):
        pass

    def collect_info(self):
        return {}

    def calculate(self, x, delay_u):
        return 0

    def get_init_u(self):
        return 0

    def get_x_init(self):
        raise RuntimeError('open loop controller has no initial state')

    def set_x_init(self, x0: np.ndarray):
        self.x_init = x0
