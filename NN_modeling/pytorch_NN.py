""" This file implements and trains the neural networks used in the black box model

"""
from typing import Union
import casadi as ca
import torch
from tqdm import trange
from torch import sigmoid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from interfaces import ResidualModel
import copy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import path_handling
import numpy as np


class SeparateMLP(torch.nn.Module):
    """ Combines multiple feed forward NN one for each output

    """

    def __init__(self, hidden_layer_sizes=(200, 200), do=0.1, mlps=None):
        super().__init__()
        if mlps is None:
            self.theta_mlp = MLP(n_out=1, hidden_layer_sizes=hidden_layer_sizes, do=do)
            self.phi_mlp = MLP(n_out=1, hidden_layer_sizes=hidden_layer_sizes, do=do)
            self.psi_mlp = MLP(n_out=1, hidden_layer_sizes=hidden_layer_sizes, do=do)
        else:
            self.theta_mlp = mlps[0]
            self.phi_mlp = mlps[1]
            self.psi_mlp = mlps[2]

    def forward(self, x):
        theta = self.theta_mlp(x)
        phi = self.phi_mlp(x)
        psi = self.psi_mlp(x)
        return torch.cat((theta, phi, psi), dim=2)


class MLP(torch.nn.Module):
    """ Feed forward NN.

    """

    def __init__(self, n_out=3, hidden_layer_sizes=(100, 100), do=0.1):
        super().__init__()
        self.first_layer = torch.nn.Linear(4, hidden_layer_sizes[0])
        self.first_do = torch.nn.Dropout(do)

        layer_list = []
        do_list = []
        for i in range(1, len(hidden_layer_sizes) - 1):
            layer_list.append(torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
            do_list.append(torch.nn.Dropout(do))
        self.hidden_layer_list = torch.nn.ModuleList(layer_list)
        self.do_layer_list = torch.nn.ModuleList(do_list)

        self.last_layer = torch.nn.Linear(hidden_layer_sizes[-1], n_out)

    def forward(self, x):
        z = sigmoid(self.first_layer(x))
        z = self.first_do(z)

        for hidden_layer, do in zip(self.hidden_layer_list, self.do_layer_list):
            z = sigmoid(hidden_layer(z))
            z = do(z)

        return self.last_layer(z)


class TrajectoryDataSet(Dataset):
    """ Dataset to store trajectory or closed loop data. (includes normalisation)

    """

    def __init__(self, data_set, mean_x=None, std_x=None, mean_y=None, std_y=None, thrust_mean=None, thrust_std=None):

        outputs = ['e_theta', 'e_phi', 'e_psi']
        inputs = ['theta', 'phi', 'psi', 'u']

        self.X = data_set[inputs].to_numpy()
        if mean_x is None:
            self.mean_x = self.X.mean(axis=0)
        else:
            self.mean_x = mean_x
        if std_x is None:
            self.std_x = self.X.std(axis=0)
        else:
            self.std_x = std_x
        self.X_norm = (self.X - self.mean_x) / self.std_x
        self.X_norm = torch.tensor(self.X_norm, dtype=torch.float)

        self.Y = data_set[outputs].to_numpy()
        if mean_y is None:
            self.mean_y = self.Y.mean(axis=0)
        else:
            self.mean_y = mean_y
        if std_y is None:
            self.std_y = self.Y.std(axis=0)
        else:
            self.std_y = std_y
        self.Y_norm = (self.Y - self.mean_y) / self.std_y
        self.Y_norm = torch.tensor(self.Y_norm, dtype=torch.float)

        self.thrust = data_set['e_thrust'].to_numpy()
        if thrust_mean is None:
            self.thrust_mean = self.thrust.mean(axis=0)
        else:
            self.thrust_mean = thrust_mean
        if thrust_std is None:
            self.thrust_std = self.thrust.std(axis=0)
        else:
            self.thrust_std = thrust_std
        self.thrust_norm = (self.thrust - self.thrust_mean) / self.thrust_std
        self.thrust_norm = torch.tensor(self.thrust_norm, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'input': self.X_norm[idx].reshape(1, -1), 'output': self.Y_norm[idx].reshape(1, -1),
                'thrust': self.thrust_norm[idx].reshape(1, -1),
                'both': torch.cat((self.Y_norm[idx], self.thrust_norm[idx].reshape(1))).reshape(1, -1),
                'theta': self.Y_norm[idx, 0].reshape(1, -1),
                'phi': self.Y_norm[idx, 1].reshape(1, -1),
                'psi': self.Y_norm[idx, 2].reshape(1, -1)}


class MyMLP(ResidualModel):
    """ NN implementation suited for casadi.

    """

    def __init__(self, mlp: MLP, mean_x, std_x, mean_y, std_y, output_name):
        self.current_MPL = mlp
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

        self.first_layer_weights = None
        self.first_layer_bias = None

        self.hidden_layer_weights = None
        self.hidden_layer_bias = None

        self.last_layer_weights = None
        self.last_layer_bias = None

        self.last_MLP = None

        self.capacity = 1000
        self.buffer_is_full = False
        self.batch_size = 64
        self.reply_counter = 0
        self.reply_buffer_inputs = np.zeros((self.capacity, len(mean_x)))
        self.reply_buffer_output = np.zeros((self.capacity, 1))

        # init weights
        self.get_current_weights()

        # init buffer
        data = pd.read_csv(path_handling.TRAINING_DATA_DIR / 'autogenerated_simulation_data.csv')
        idx = min(self.capacity, len(data))
        self.reply_counter = idx
        random_indices = np.arange(len(data))
        np.random.shuffle(random_indices)
        self.reply_buffer_inputs[:idx] = data[['theta', 'phi', 'psi', 'u']].to_numpy()[random_indices][:idx]
        self.reply_buffer_output[:idx] = data[[output_name]].to_numpy()[random_indices][:idx]
        self.reply_buffer_inputs = torch.tensor(self.reply_buffer_inputs, dtype=torch.float)
        self.reply_buffer_output = torch.tensor(self.reply_buffer_output, dtype=torch.float)

    def get_current_weights(self):
        self.first_layer_weights = self.current_MPL.first_layer.weight.detach().numpy()
        self.first_layer_bias = self.current_MPL.first_layer.bias.detach().numpy()

        self.hidden_layer_weights = []
        self.hidden_layer_bias = []
        for hidden_layer in self.current_MPL.hidden_layer_list:
            self.hidden_layer_weights.append(hidden_layer.weight.detach().numpy())
            self.hidden_layer_bias.append(hidden_layer.bias.detach().numpy())

        self.last_layer_weights = self.current_MPL.last_layer.weight.detach().numpy()
        self.last_layer_bias = self.current_MPL.last_layer.bias.detach().numpy()

    def predict(self, x: Union[ca.MX, np.ndarray], u: Union[ca.MX, float]) -> Union[ca.MX, np.ndarray, float]:
        if isinstance(x, ca.casadi.MX) or isinstance(u, ca.casadi.MX):
            mpl_input = ca.vertcat(x, u)
            current_exp = ca.exp

        else:
            mpl_input = x.copy()
            mpl_input = np.append(mpl_input, u)
            current_exp = np.exp

        my_sigmoid = lambda temp: 1 / (1 + current_exp(-temp))
        mpl_input = (mpl_input - self.mean_x) / self.std_x
        z = my_sigmoid(self.first_layer_weights @ mpl_input + self.first_layer_bias)
        for hidden_weight, hidden_bias in zip(self.hidden_layer_weights, self.hidden_layer_bias):
            z = my_sigmoid(hidden_weight @ z + hidden_bias)

        return (self.last_layer_weights @ z + self.last_layer_bias) * self.std_y + self.mean_y

    def predictive_mean(self, X):
        X_norm = (X - self.mean_x) / self.std_x
        my_sigmoid = lambda temp: 1 / (1 + np.exp(-temp))

        z = my_sigmoid(self.first_layer_weights @ X_norm.transpose() + self.first_layer_bias.reshape(-1, 1))
        for hidden_weight, hidden_bias in zip(self.hidden_layer_weights, self.hidden_layer_bias):
            z = my_sigmoid(hidden_weight @ z + hidden_bias.reshape(-1, 1))

        return (self.last_layer_weights @ z + self.last_layer_bias.reshape(-1, 1)) * self.std_y.reshape(-1,
                                                                                                        1) + self.mean_y.reshape(
            -1, 1)

    def predictive_variance(self, X):
        return np.zeros((self.last_layer_weights.shape[0], len(X), len(X)))

    def predict_variance(self, x: Union[ca.MX, np.ndarray], u: Union[ca.MX, float]) -> float:
        return 0

    def update(self, x: np.ndarray, u: float, y: np.ndarray, rule_one_activated: np.ndarray):
        self.last_MLP = copy.deepcopy(self.current_MPL)

        # normalize in-/ and output
        mlp_input = np.concatenate((x, np.array([u]))).reshape(1, -1)
        mlp_input = (mlp_input - self.mean_x) / self.std_x
        mlp_input = torch.tensor(mlp_input, dtype=torch.float)

        y_gt = torch.tensor((y - self.mean_y) / self.std_y, dtype=torch.float)

        # include new data point in reply buffer
        if self.reply_counter == self.capacity - 1:
            self.buffer_is_full = True
        self.reply_counter = (self.reply_counter + 1) % self.capacity
        self.reply_buffer_inputs[self.reply_counter] = mlp_input
        self.reply_buffer_output[self.reply_counter] = y_gt

        self.current_MPL.train()

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(self.current_MPL.parameters(), lr=1e-5)

        # get batch
        N = self.capacity - 1 if self.buffer_is_full else self.reply_counter
        batch_indices = np.random.randint(0, N + 1, size=300)
        batch_input = self.reply_buffer_inputs[batch_indices].reshape(300, 1, -1)
        batch_output = self.reply_buffer_output[batch_indices].reshape(300, 1, -1)

        # Compute loss
        y_pred = self.current_MPL(batch_input)
        loss = criterion(y_pred, batch_output)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.current_MPL.eval()
        self.get_current_weights()

    def undo_update(self):
        self.current_MPL = self.last_MLP
        self.get_current_weights()

    def save(self, name):
        with open(path_handling.RESIDUAL_Model_DIR / (name + '.pkl'), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


class MySeparateMLP(ResidualModel):
    """ combines multiple feed forward NN suited for casadi.

    """

    def __init__(self, separate_mlp, mean_x, std_x, mean_y, std_y):
        self.my_theta_mlp = MyMLP(separate_mlp.theta_mlp, mean_x, std_x, mean_y[0], std_y[0], output_name='e_theta')
        self.my_phi_mlp = MyMLP(separate_mlp.phi_mlp, mean_x, std_x, mean_y[1], std_y[1], output_name='e_phi')
        self.my_psi_mlp = MyMLP(separate_mlp.psi_mlp, mean_x, std_x, mean_y[2], std_y[2], output_name='e_psi')
        self.separate_mlp = separate_mlp

        self.last_updated = None

    def predict(self, x: Union[ca.MX, np.ndarray], u: Union[ca.MX, float]) -> Union[ca.MX, np.ndarray, float]:
        theta = self.my_theta_mlp.predict(x, u)
        phi = self.my_phi_mlp.predict(x, u)
        psi = self.my_psi_mlp.predict(x, u)

        if isinstance(x, ca.casadi.MX) or isinstance(u, ca.casadi.MX):
            return ca.vertcat(theta, phi, psi)

        else:
            return np.concatenate((theta, phi, psi))

    def predictive_mean(self, X):
        theta = self.my_theta_mlp.predictive_mean(X)
        phi = self.my_phi_mlp.predictive_mean(X)
        psi = self.my_psi_mlp.predictive_mean(X)

        return np.vstack((theta, phi, psi))

    def predictive_variance(self, X):
        return np.zeros((3, len(X), len(X)))

    def predict_variance(self, x: Union[ca.MX, np.ndarray], u: Union[ca.MX, float]) -> float:
        return 0

    def update(self, x: np.ndarray, u: float, y: np.ndarray, rule_one_activated: np.ndarray):
        self.last_updated = rule_one_activated
        if rule_one_activated[0]:
            self.my_theta_mlp.update(x, u, y[0:1], rule_one_activated)
        if rule_one_activated[1]:
            self.my_theta_mlp.update(x, u, y[1:2], rule_one_activated)
        if rule_one_activated[2]:
            self.my_theta_mlp.update(x, u, y[2:], rule_one_activated)

    def undo_update(self):
        if self.last_updated[0]:
            self.my_theta_mlp.undo_update()
        if self.last_updated[1]:
            self.my_theta_mlp.undo_update()
        if self.last_updated[2]:
            self.my_theta_mlp.undo_update()

    def save(self, name):
        with open(path_handling.RESIDUAL_Model_DIR / (name + '.pkl'), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


def load_model(name: str) -> ResidualModel:
    """ Loads an existing NN from memory

    Parameters
    ----------
    name : str
        name of the NN (residual model name)

    Returns
    -------
    ResidualModel
        the residual model with the given name

    """

    class CustomUnpickler(pickle.Unpickler):

        def find_class(self, module, class_name):
            if class_name == 'MyMLP':
                return MyMLP
            if class_name == 'MLP':
                return MLP
            if class_name == 'SeparateMLP':
                return SeparateMLP
            if class_name == 'MySeparateMLP':
                return MySeparateMLP
            return super().find_class(module, class_name)

    loadedMLP = CustomUnpickler(open(path_handling.RESIDUAL_Model_DIR / (name + '.pkl'), 'rb')).load()

    return loadedMLP


def _train(model, train_dl, val_dl, output_name, epochs=1000):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())

    early_sop_after = 10
    no_improvement_since = 0

    min_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    training_loss_list = [np.nan] * epochs
    validation_loss_list = [np.nan] * epochs
    pbar = trange(epochs, unit="Epochs")
    for i in pbar:
        model.train()
        train_loss = 0
        for batch in train_dl:
            model_out = model(batch['input'])
            loss = criterion(model_out, batch[output_name])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl)
        training_loss_list[i] = train_loss

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dl:
                model_out = model(batch['input'])
                loss = criterion(model_out, batch[output_name])
                val_loss += loss.item()
            val_loss = val_loss / len(val_dl)
        if min_valid_loss > val_loss:
            no_improvement_since = 0
            min_valid_loss = val_loss
            best_model = copy.deepcopy(model)
        else:
            no_improvement_since += 1

        if no_improvement_since >= early_sop_after:
            break

        validation_loss_list[i] = val_loss

        pbar.set_postfix(training_loss=train_loss, validation_loss=val_loss)
    pbar.close()
    if no_improvement_since >= early_sop_after:
        print('early stopped since no further improvement.')

    return best_model, np.array(training_loss_list), np.array(validation_loss_list)


def create_residual_MLP_model(save_name: str, train_data: pd.DataFrame, val_data: pd.DataFrame, epochs: int = 150,
                              visualize=False):
    """ crates, trains and tests the residual model

    Parameters
    ----------
    save_name : str
        name of the residual model
    train_data : pd.DataFrame
        training data (use data_generation/simulation_data_generation.py to create this DataFrame)
    val_data : pd.DataFrame
        validation data
    epochs : int
        number of training epochs
    visualize : bool
        whether result should be visualized

    Returns
    -------

    """
    train_data_set = TrajectoryDataSet(train_data)
    train_data_loader = DataLoader(train_data_set, batch_size=128, shuffle=True)

    val_data_set = TrajectoryDataSet(val_data, train_data_set.mean_x, train_data_set.std_x,
                                     train_data_set.mean_y, train_data_set.std_y)
    val_data_loader = DataLoader(val_data_set, batch_size=128, shuffle=False)

    theta_model = MLP(n_out=1, hidden_layer_sizes=[100], do=0.1)
    phi_model = MLP(n_out=1, hidden_layer_sizes=[100], do=0.1)
    psi_model = MLP(n_out=1, hidden_layer_sizes=[100], do=0.1)
    model_thrust = MLP(n_out=1, hidden_layer_sizes=[100], do=0.1)

    print('train NN for delta theta...')
    theta_model, train_loss1, val_loss1 = _train(theta_model, train_data_loader, val_data_loader, 'theta', epochs)
    print('train NN for delta phi...')
    phi_model, train_loss2, val_loss2 = _train(phi_model, train_data_loader, val_data_loader, 'phi', epochs)
    print('train NN for delta psi...')
    psi_model, train_loss3, val_loss3 = _train(psi_model, train_data_loader, val_data_loader, 'psi', epochs)
    model = SeparateMLP(mlps=[theta_model, phi_model, psi_model])
    print('train NN for delta thrust...')
    model_thrust, train_loss_thrust, val_loss_thrust = _train(model_thrust, train_data_loader, val_data_loader,
                                                              'thrust',
                                                              epochs)
    fig = plt.figure('training_history')
    plt.plot(train_loss1, label='trainloss1')
    plt.plot(train_loss2, label='trainloss2')
    plt.plot(train_loss3, label='trainloss3')
    plt.plot(val_loss1, label='valloss1')
    plt.plot(val_loss1, label='valloss2')
    plt.plot(val_loss1, label='valloss3')
    plt.plot(train_loss_thrust, label='trainlossthrust')
    plt.plot(val_loss_thrust, label='vallossthrust')
    plt.legend()
    if visualize:
        plt.show()
    fig.savefig(path_handling.RESIDUAL_Model_DIR / 'train_history.png')

    history_df = pd.DataFrame({
        'theta_loss_train': train_loss1,
        'phi_loss_train': train_loss2,
        'psi_loss_train': train_loss3,
        'theta_loss_val': val_loss1,
        'phi_loss_val': val_loss2,
        'psi_loss_val': val_loss3,
        'thrust_loss_train': train_loss_thrust,
        'thrust_loss_val': val_loss_thrust,
    })
    history_df.to_csv(path_handling.RESIDUAL_Model_DIR / 'train_history.csv')
    model.eval()

    my_mlp = MySeparateMLP(model, train_data_set.mean_x, train_data_set.std_x, train_data_set.mean_y,
                           train_data_set.std_y)
    my_mlp.save(save_name)

    my_mlp_thrust = MyMLP(model_thrust, train_data_set.mean_x, train_data_set.std_x, train_data_set.thrust_mean,
                          train_data_set.thrust_std, output_name='e_thrust')
    my_mlp_thrust.save(save_name + '_thrust')
    print('saved.')

    plant_data_set = pd.read_csv(path_handling.TEST_DATA_DIR / 'test.csv')
    test_data_set = TrajectoryDataSet(plant_data_set, train_data_set.mean_x, train_data_set.std_x,
                                      train_data_set.mean_y, train_data_set.std_y)
    e_model = model(test_data_set.X_norm.reshape(1, -1, 4)).detach().numpy().reshape(-1, 3)

    error_after_correction = -e_model + test_data_set.Y_norm.numpy()
    mse_after_correction = np.linalg.norm(error_after_correction, axis=1)

    mse_before_correction = np.linalg.norm(test_data_set.Y_norm.numpy(), axis=1)

    names = ["theta", "phi", "psi"]
    for i in range(3):
        fig = plt.figure(f'prediction_error_{names[i]}')
        plt.plot(test_data_set.Y_norm[:, i], label='before')
        plt.plot(error_after_correction[:, i], label='after', alpha=0.6)
        plt.title(f'{names[i]} error')
        plt.legend()
        fig.savefig(path_handling.RESIDUAL_Model_DIR / f'test_error_{names[i]}.png')
        if visualize:
            plt.show()
        plt.close(fig)
    # print(f'prior mse: {mse_before_correction}')
    # print(f'posterior mse: {mse_after_correction}')
    # a = my_mlp.predictive_mean(test_data_set.X).transpose()
    # correct = np.allclose(a, e_model * train_data_set.std_y + train_data_set.mean_y, atol=1e-6)
    # x_sample, u_sample = test_data_set.X[0, 0:3], test_data_set.X[0, 3]
    # b = my_mlp.predict(x_sample, u_sample)
    # x_sym, u_sym = ca.MX.sym('x', (3, 1)), ca.MX.sym('u')
    # b_sym = my_mlp.predict(x_sym, u_sym)
    # get_b = ca.Function('get_b', [x_sym, u_sym], [b_sym])
    # b_2 = get_b(x_sample, u_sample).full()
    # print()
