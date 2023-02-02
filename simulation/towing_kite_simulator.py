from simulation import simulator
import model_zoo
from controller.basic_contoller import Controller
import numpy as np
from typing import List
from typing import Callable, Union


def get_kite_simulator(controller: Controller = None, noise_std_deg: Union[List, np.ndarray] = None, use_physical=False,
                       wind_function: Callable[[float], float] = None, plant_number: int = 1,
                       drift_function: Callable[[float], float] = None):
    """
    crates the simulator for the plant towing kite model
    :param controller: Controller used to compute u
    :param noise_std_deg: list of the standard deviation of the noise in degree
    :param use_physical: whether to use the physical model
    :param wind_function: see model_zoo.get_erhard_dgl_and_thrust_function
    :param plant_number: number of plant parameter_set
    :param drift_function: see model_zoo.get_erhard_dgl_and_thrust_function
    :return: simulator object
    """
    if use_physical:
        parameters = model_zoo.ERHARD_PHYSICAL_PARAMETER
    elif plant_number == 1:
        parameters = model_zoo.ERHARD_PLANT_PARAMETER
    elif plant_number == 2:
        parameters = model_zoo.ERHARD_PLANT_PARAMETER_2
    elif plant_number == 3:
        parameters = model_zoo.ERHARD_PLANT_PARAMETER_3
    else:
        raise RuntimeError('unknown plant number')
    dynamic_function, thrust_function = model_zoo.get_erhard_dgl_and_thrust_function(parameters,
                                                                                     wind_function=wind_function,
                                                                                     drift_function=drift_function)
    constraints = model_zoo.KITE_CONSTRAINTS
    Ts = model_zoo.ERHARD_Ts

    if noise_std_deg is not None:
        def noise_function() -> np.ndarray:
            noise = np.random.randn(len(noise_std_deg)) * np.array(noise_std_deg) * np.pi / 180
            return noise
    else:
        noise_function = None

    sim = simulator.Simulator(dynamic_function, Ts, np.array(constraints['u_min']), np.array(constraints['u_max']),
                              controller=controller, value_function=thrust_function, noise_function=noise_function)
    return sim
