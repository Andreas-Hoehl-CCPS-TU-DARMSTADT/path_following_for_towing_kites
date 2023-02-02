import pandas
import path_handling
import model_zoo
import pandas as pd
import numpy as np
from pathlib import Path


def extract_data_from_simulation(save_name: str, save_folder: Path, simulation_folder: Path):
    """extracts a dataset from all simulations within the given folder name

    stores the data of the simulations located at simulation_folder in save_folder/save_name.csv.
    The dataset contains: 't', 'theta', 'phi', 'psi', 'u' ,'e_theta', 'e_phi', 'e_psi', 'e_thrust',
    where e means the difference between the 1-step prediction and the measurement of the next step
    (measurement-prediction).

    Parameters
    ----------
    save_name : str
        name of the output csv file
    save_folder: Path
        path of the output csv file
    simulation_folder:
        path of the input simulations

    Returns
    -------

    """
    path = path_handling.SIMULATION_DIR / simulation_folder
    physical_prediction_model = model_zoo.get_Erhard_prediction_model('physical', steps=1, discretization_method='rk4')
    select = ['t', 'integrated_thrust_per_time_step', 'u', 'theta_measure', 'phi_measure', 'psi_measure',
              'theta_1_step_prediction', 'phi_1_step_prediction', 'psi_1_step_prediction']

    all_data_df = pd.DataFrame()
    for simulation_path in path.iterdir():
        if not simulation_path.is_dir():
            continue

        simulation_df = pd.read_csv(simulation_path / 'result.csv')[select]

        measurements = simulation_df[['theta_measure', 'phi_measure', 'psi_measure']].to_numpy()
        predictions = simulation_df[
            ['theta_1_step_prediction', 'phi_1_step_prediction', 'psi_1_step_prediction']].to_numpy()
        inputs = simulation_df['u'].to_numpy()
        thrust_integral_measurements = simulation_df['integrated_thrust_per_time_step'].to_numpy()
        thrust_integral_measurements += + np.random.randn(
            len(thrust_integral_measurements)) * model_zoo.ERHARD_NOISE_THRUST_MEDIUM  # add noise
        thrust_prediction = np.array(
            [physical_prediction_model.predict_thrust(-1, measurements[i], inputs[i]) for i in range(len(inputs) - 1)])

        state_error = measurements[1:] - predictions[:-1]
        thrust_error = thrust_integral_measurements[:-1] - thrust_prediction
        data_df = pandas.DataFrame(
            {'t': simulation_df['t'][:-1], 'theta': measurements[:-1, 0], 'phi': measurements[:-1, 1],
             'psi': measurements[:-1, 2], 'u': inputs[:-1],
             'e_theta': state_error[:, 0], 'e_phi': state_error[:, 1], 'e_psi': state_error[:, 2],
             'e_thrust': thrust_error})
        all_data_df = pd.concat((all_data_df, data_df))

    all_data_df.to_csv(save_folder / f'{save_name}.csv', index=False)
