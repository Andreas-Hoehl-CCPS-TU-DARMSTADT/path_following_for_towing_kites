from abc import abstractmethod
import casadi as ca
import numpy as np
from typing import TypedDict, List, Dict, Union, Optional


class ResidualModel:
    @abstractmethod
    def predict(self, x: Union[ca.MX, np.ndarray], u: Union[ca.MX, float]) -> Union[ca.MX, np.ndarray, float]:
        """ predicts the model plant error.

        :param x: state at time k
        :param u: input at time k
        :return: predicted model plant miss match at time k+1 this may also be the residual thrust
        """

    @abstractmethod
    def predict_variance(self, x: Union[ca.MX, np.ndarray], u: Union[ca.MX, float]) -> float:
        """ returns the prediction variance of the model

        :param x: state at time k
        :param u: input at time k
        :return: variance of the predicted model plant miss match at time k+1
        """

    @abstractmethod
    def update(self, x: np.ndarray, u: float, y: np.ndarray, rule_one_activated: np.ndarray):
        """ update the residual model with a new data point

        :param x: current state
        :param u: current input
        :param y: desired output
        :param rule_one_activated: says for which output the rule one from Michael's diss is activated
        """

    @abstractmethod
    def undo_update(self):
        """
        reverts the last update
        """


class SimulationResult(TypedDict):
    x: np.ndarray
    u: np.ndarray
    u_controller: np.ndarray
    t: np.ndarray
    k_index: np.ndarray
    info: List[Dict]
    integrate_value: float
    integrate_time_step_value: np.ndarray
    moment_values: np.ndarray
    measurements: np.ndarray
    computation_times: np.ndarray


class SimulationParameter(TypedDict):
    initial_value: Optional[List[float]]  # if None the initial state of the reference of the controller is used.
    simulation_time: float  # total simulation time
    noise_std_deg: List  # std dev in deg. Set None to deactivate noise.
    delay_u: bool  # if True means that only measurements up to k-1 are available to the controller
    wind_function_index: int  # index of the wind function
    wind_frequency: float  # frequency for first wind function
    magnitude: float  # magnitude of the wind function (relative to constant wind)
    delay: float  # delay time for second wind function
    tau: float  # steepness for second wind function
    drift_function_index: int  # index of the drift function
    plant_number: int  # index of the plant that is used


class OnlineLearningSettings(TypedDict):
    active_algorithm: int  # online learning method
    monitor_algorithm: List[int]  # calculate and monitor update algorithms given in list
    n_max_data_points: int
    keep_offline_data: bool
    threshold_x_err: List[float]
    threshold_variance: List[float]


class TTSettings(TypedDict):
    hmin: Optional[float]
    Q: List[List[float]]
    final_weight: float
    R: float
    S: float


class PFSettings(TypedDict):
    hmin: Optional[float]  # use None to deactivate height constraint
    Q: List[List[float]]
    final_weight: float
    S: List[List[float]]
    n_virtual: int
    virtual_state_lb: List[float]
    virtual_state_ub: List[float]
    virtual_input_lb: float
    virtual_input_ub: float


class ControlParameter(TypedDict):
    use_soft_constraint: bool  # whether height constraint is soft
    time_variant: bool  # wind function is known to the controller
    wind_function: int  # wind function of the controller (only applied if time_variant is True)
    type: str  # pf or tt or uf (for uf only reference is of interest)
    prediction_horizon: int

    prediction_model: str  # physical hybrid or plant
    residual_model: Optional[str]  # GP_high_statemlp_test

    reference: str  # name of the trajectory that is used as reference

    online_learning_settings: Optional[OnlineLearningSettings]

    discretization_method: str  # direct_multiple_shooting or direct_single_shooting
    integration_method: str  # rk4 or euler
    integration_steps: int  # number of integration steps per time step

    pf_parameter: Optional[PFSettings]  # parameter only for path following
    tt_parameter: Optional[TTSettings]  # parameter only for trajectory tracking
