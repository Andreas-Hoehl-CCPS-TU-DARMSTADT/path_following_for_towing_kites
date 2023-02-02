"""Implements the trajectory tracking and path following MPC controller.

Use controller/controller_zoo.py to instantiate the controller with the desired parameter settings.

This file can be should be imported as a module and contains the following
classes:
    * MPC - abstract class from which both controllers inherit
    * PathFollowingMPC - implementation of the path following MPC
    * TrajectoryTrackingMPC - implementation of the trajectory tracking controller
"""
import numpy as np
from model_zoo import PredictionModel
from model_zoo import ExtendedPredictionModel
import utils
import casadi as ca
from controller.basic_contoller import Controller
from typing import Tuple, Union, Optional
from utils import ControlPath
from abc import abstractmethod
from copy import deepcopy
import interfaces


class MPC(Controller):
    def __init__(self, Q: np.ndarray, S: Optional[np.ndarray], final_weight: float, use_soft_constraint: bool,
                 prediction_model: PredictionModel, N: int, constraints: utils.NLPConstraints, discretization: str,
                 online_learning_options=Optional[interfaces.OnlineLearningSettings]):
        """

        Parameters
        ----------
        Q : np.ndarray
            weight matrix for quadratic cost on states
        S : Optional[np.ndarray]
            cost on input changes (not applied if None)
        final_weight : float
            weight of the terminal cost
        use_soft_constraint : bool
            whether height constraint should be soft using slack variables
        prediction_model : PredictionModel
            prediction model of the controller
        N : int
            prediction horizon
        constraints : utils.NLPConstraints
            constraints of the OCP
        discretization : str (default='direct_multiple_shooting')
            discretization method of the optimal control problem use either
            1) 'direct_multiple_shooting' or
            2) 'direct_single_shooting'
        online_learning_options : Optional[interfaces.OnlineLearningSettings]
            defines the option used for online learning (see RGPMPC from Michael Maiworm for more details.)
            online learning will be deactivated if None
        """
        self.prediction_model = prediction_model
        self.N = N
        self.Q = Q
        self.S = S
        self.final_weight = final_weight
        self.discretization = discretization
        self.constrains = constraints
        self.use_soft_constraint = use_soft_constraint

        # contains the last solution the complete prediction horizon
        self.t = 0
        self.k_offset = 0  # offset of the reference trajectory
        self.last_u = None
        self.last_x = None
        self.last_loss = None
        self.last_sol = None
        self.success = True

        # this is used for initial guesses and to collect info
        self.state_posterior = None
        self.output_posterior = None

        # this is used to store the solutions of algorithm 3
        self.prior_solution3 = None
        self.posterior_solution3 = None

        # this is used to keep the old NLP and prediction model for algorithm 2
        self.NLP_old = None

        # recursive update information
        self.updated = False
        self.last_prediction_error = None
        self.last_prediction_variance = None
        # prior and posterior loss algorithm 1, 2, 3
        self.prior_loss1 = None
        self.posterior_loss1 = None
        self.prior_loss2 = None
        self.posterior_loss2 = None
        self.prior_loss3 = None
        self.posterior_loss3 = None
        self.success1 = None
        self.success2 = None
        self.success3 = None

        self.active_algorithm = online_learning_options['active_algorithm'] if online_learning_options else None
        self.monitor_algorithm = online_learning_options['monitor_algorithm'] if online_learning_options else None
        self.x_threshold = np.array(online_learning_options['threshold_x_err']) if online_learning_options else None
        self.var_threshold = np.array(
            online_learning_options['threshold_variance']) if online_learning_options else None

        self.NLP = self._create_NLP()

        self.cost_options = None

    @abstractmethod
    def _stage_cost(self, x: ca.MX, u: ca.MX, k: int, **kwargs) -> ca.MX:
        pass

    @abstractmethod
    def _terminal_cost(self, x: ca.MX, **kwargs) -> ca.MX:
        pass

    def _delta_u_cost(self, u: ca.MX):

        if self.S is not None:
            delta_u = u[1:, :] - u[:-1, :]
            last_output = self.last_v if isinstance(self, PathFollowingMPC) else self.last_u
            if last_output is not None:
                delta_u_0 = u[0, :] - last_output[0].reshape(1, -1)
                loss = delta_u_0 @ self.S @ delta_u_0.T
            else:
                loss = 0

            for i in range(self.N - 1):
                loss += delta_u[i, :] @ self.S @ delta_u[i, :].T

            return loss
        else:
            return 0

    def _create_NLP(self) -> utils.ShootingNLP:
        if self.discretization == 'direct_single_shooting':
            NLP = utils.DirectSingleShootingNLP(self.N, self.prediction_model, self._stage_cost, self._terminal_cost,
                                                self._delta_u_cost, self.constrains, self.use_soft_constraint)
        elif self.discretization == 'direct_multiple_shooting':
            NLP = utils.DirectMultipleShootingNLP(self.N, self.prediction_model, self._stage_cost, self._terminal_cost,
                                                  self._delta_u_cost, self.constrains, self.use_soft_constraint)
        else:
            raise RuntimeError('unknown discretization')

        return NLP

    def reset(self) -> None:
        """ resets the controller.

        Returns
        -------

        """
        self.last_u = None
        self.last_x = None
        self.last_loss = None
        self.last_sol = None
        self.t = 0

    @abstractmethod
    def _get_new_ground_truth(self, x_new: np.ndarray):
        """ calculates the new ground truth for the prediction model update.

        This method is only used if online learning is active.

        Parameters
        ----------
        x_new : np.ndarray
            xk+1

        Returns
        -------
        np.ndarray
            yk
        """

    def _algorithm1(self):
        """ online updating algorithm 1.

        See master Theses of Andreas Höhl for more details.

        Returns
        -------
        Tuple[float, float, bool]
            prior_loss, loss, success
        """
        prior_loss = self.last_loss
        if isinstance(self, PathFollowingMPC):
            z_old = self.last_z[0]
            states, output, loss, sol, success = self.NLP.solve(self.t - self.prediction_model.Ts, z_old, self.last_z,
                                                                self.last_v, self.cost_options)
        elif isinstance(self, TrajectoryTrackingMPC):
            x_old = self.last_x[0]
            self.index = self.index - 1 if self.index > 0 else len(self.u_ref) - 1  # revert index to get old ref
            states, output, loss, sol, success = self.NLP.solve(self.t - self.prediction_model.Ts, x_old, self.last_x,
                                                                self.last_u, self.cost_options)
            self.index = (self.index + 1) % len(self.u_ref)  # do index step again to get current reference
        else:
            raise RuntimeError('unknown mpc')

        if self.active_algorithm == 1:
            self.state_posterior = states
            self.output_posterior = output
        else:
            raise RuntimeError('wrong Algorithm for online updates')
        return prior_loss, loss, success

    def _algorithm2(self):
        """ online updating algorithm 2.

        See master Theses of Andreas Höhl for more details.

        Returns
        -------
        Tuple[float, float, bool]
            prior_loss, loss, success
        """
        prior_loss = self.last_loss
        if isinstance(self, PathFollowingMPC):
            posterior_loss, success = self.NLP.get_cost(self.last_z[0], self.last_v)
        elif isinstance(self, TrajectoryTrackingMPC):
            self.index = self.index - 1 if self.index > 0 else len(self.u_ref) - 1  # revert index to get old ref
            posterior_loss, success = self.NLP.get_cost(self.last_x[0], self.last_u)
            self.index = (self.index + 1) % len(self.u_ref)  # do index step again to get current reference
        else:
            raise RuntimeError('unknown mpc')
        return prior_loss, posterior_loss, success

    def _algorithm3(self, x_new):
        """ online updating algorithm 3.

        See master Theses of Andreas Höhl for more details.

        Returns
        -------
        Tuple[float, float, bool]
            prior_loss, loss, success
        """

        if isinstance(self, PathFollowingMPC):
            z0, v0 = self._get_initial_guess()
            z_init = np.concatenate((x_new, self.current_virtual_state))
            self.prior_solution3 = self.NLP_old.solve(self.t, z_init, z0, v0, self.cost_options)
            self.posterior_solution3 = self.NLP.solve(self.t, z_init, z0, v0, self.cost_options)
        elif isinstance(self, TrajectoryTrackingMPC):
            x0, u0 = self._get_initial_guess()
            self.prior_solution3 = self.NLP_old.solve(self.t, x_new, x0, u0, self.cost_options)
            self.posterior_solution3 = self.NLP.solve(self.t, x_new, x0, u0, self.cost_options)
        else:
            raise RuntimeError('unknown mpc')

        if self.active_algorithm == 3:
            self.state_posterior = self.posterior_solution3[0]
            self.output_posterior = self.posterior_solution3[1]
        return self.prior_solution3[2], self.posterior_solution3[2], self.posterior_solution3[4]

    # only use rule 1
    def _algorithm0(self):
        """ ignore RRPMPC and update in every time step

        See master Theses of Andreas Höhl for more details.

        Returns
        -------
        Tuple[float, float, bool]
            prior_loss, loss, success
        """
        if self.active_algorithm == 0:
            if isinstance(self, PathFollowingMPC):
                self.state_posterior = self.last_z
                self.output_posterior = self.last_v
            elif isinstance(self, TrajectoryTrackingMPC):
                self.state_posterior = self.last_x
                self.output_posterior = self.last_u
        return 1, 0, True

    def _update(self, x_new: np.ndarray):
        """ updates the prediction model if online learning is activated.

        Parameters
        ----------
        x_new : np.ndarray
            new measurement

        Returns
        -------

        """

        updated = False

        self.last_prediction_error = x_new - self.last_x[1]
        self.last_prediction_variance = self.prediction_model.residual_state_model.predict_variance(self.last_x[0],
                                                                                                    self.last_u[0])
        threshold_is_exceeded = np.logical_or(self.x_threshold <= np.abs(self.last_prediction_error),
                                              self.var_threshold <= self.last_prediction_variance)
        if threshold_is_exceeded.any():
            # rule one is true now update
            self.NLP_old = deepcopy(self.NLP)
            y = self._get_new_ground_truth(x_new)
            self.prediction_model.residual_state_model.update(self.last_x[0], self.last_u[0], y, threshold_is_exceeded)
            self.NLP = self._create_NLP()

            if self.active_algorithm == 1 or 1 in self.monitor_algorithm:
                self.prior_loss1, self.posterior_loss1, self.success1 = self._algorithm1()
            if self.active_algorithm == 2 or 2 in self.monitor_algorithm:
                self.prior_loss2, self.posterior_loss2, self.success2 = self._algorithm2()
            if self.active_algorithm == 3 or 3 in self.monitor_algorithm:
                self.prior_loss3, self.posterior_loss3, self.success3 = self._algorithm3(x_new)

            if self.active_algorithm == 1:
                posterior_loss = self.posterior_loss1
                prior_loss = self.prior_loss1
                success = self.success1
            elif self.active_algorithm == 2:
                posterior_loss = self.posterior_loss2
                prior_loss = self.prior_loss2
                success = self.success2
            elif self.active_algorithm == 3:
                posterior_loss = self.posterior_loss3
                prior_loss = self.prior_loss3
                success = self.success3
            else:
                posterior_loss, prior_loss, success = self._algorithm0()

            if posterior_loss > prior_loss or not success:
                # rule two is not fulfilled -> revert update
                #  self.NLP = self.NLP_old  # causes recursion error
                self.prediction_model.residual_state_model.undo_update()
                self.NLP = self._create_NLP()
                self.NLP_old = None
            else:
                # rule two is fulfilled -> keep update
                updated = True
        else:
            # rule one is not fulfilled
            self.prior_loss1 = None
            self.posterior_loss1 = None
            self.prior_loss2 = None
            self.posterior_loss2 = None
            self.prior_loss3 = None
            self.posterior_loss3 = None
            self.success1 = None
            self.success2 = None
            self.success3 = None
            self.posterior_solution3 = None
            self.prior_solution3 = None

        self.updated = updated

    def collect_info(self):
        """ Used by the simulator to collect internal information of the controller

        Returns
        -------
        dict
            containing internal controller info such as predicted states and inputs and so on.
        """
        if self.last_x is None:
            return {}
        else:
            return {'x_sol': self.last_x,
                    'u_sol': self.last_u,
                    'loss': self.last_loss,
                    'updated': self.updated,
                    'prediction_error': self.last_prediction_error,
                    'prediction_variance': self.last_prediction_variance,
                    'prior_loss1': self.prior_loss1,
                    'posterior_loss1': self.posterior_loss1,
                    'prior_loss2': self.prior_loss2,
                    'posterior_loss2': self.posterior_loss2,
                    'prior_loss3': self.prior_loss3,
                    'posterior_loss3': self.posterior_loss3,
                    'state_posterior': self.state_posterior,
                    'output_posterior': self.output_posterior,
                    'success': self.success}


class PathFollowingMPC(MPC):

    def __init__(self, Q: np.ndarray, S: Union[np.ndarray, None], final_weight: float, use_soft_constraint: bool,
                 extended_prediction_model: ExtendedPredictionModel,
                 N: int, constraints: utils.NLPConstraints, path: ControlPath,
                 discretization: str = 'direct_multiple_shooting',
                 online_learning_options=Optional[interfaces.OnlineLearningSettings]):
        """ Path following MPC implementation.

        Ideally, use controller/controller_zoo.py to initialize it.

        Parameters
        ----------
        Q : np.ndarray
            weighting matrix for the state penalty
        S : np.ndarray
            weighting matrix for input changes (this also includes the virtual input)
        final_weight : float
            weighs the terminal cost
        use_soft_constraint : bool
            whether soft constrains via slack variables should be used
        extended_prediction_model : ExtendedPredictionModel
            prediction model of the path following controller including the virtual system
        N : int
            prediction horizon
        constraints : utils.NLPConstraints
            constraints of the system (also including the virtual system)
        path : ControlPath
            reference Path
        discretization : str (default='direct_multiple_shooting')
            discretization method of the optimal control problem use either
            1) 'direct_multiple_shooting' or
            2) 'direct_single_shooting'
        online_learning_options : Optional[interfaces.OnlineLearningSettings]
            defines the option used for online learning (see RGPMPC from Michael Maiworm for more details.)
            online learning will be deactivated if None
        """

        super().__init__(Q, S, final_weight, use_soft_constraint, extended_prediction_model, N, constraints,
                         discretization, online_learning_options)

        self.prediction_model = extended_prediction_model  # just that the ide recognizes the subtype
        self.last_v = None
        self.last_z = None
        self.current_virtual_state = None
        self.path = path
        self.cost_options = {'nx': self.prediction_model.nx,
                             'path': self.path}

    def _stage_cost(self, z, _, __, **kwargs) -> ca.MX:
        path_parameter = z[kwargs['nx']]
        x_error = z[:kwargs['nx']] - kwargs['path'](path_parameter)

        return x_error.T @ self.Q @ x_error

    def _terminal_cost(self, z: ca.MX, **kwargs) -> ca.MX:
        path_parameter = z[kwargs['nx']]
        x_error = z[:kwargs['nx']] - kwargs['path'](path_parameter)

        return self.final_weight * x_error.T @ self.Q @ x_error

    def reset(self):
        """ resets the path following controller

        Returns
        -------

        """
        super().reset()
        self.last_v = None
        self.last_z = None
        self.current_virtual_state = None

    def get_x_init(self):
        """ returns the initial state of the controller (i.e. first state of reference)

        Returns
        -------
        np.ndarray
            initial state of the reference

        """
        return self.path.trajectory.x[0]

    def set_x_init(self, x0: np.ndarray):
        """ sets the initial state of the controller.

        sets the path parameter to the value with minimizes the distance between the reference and the given
        initial state.

        Parameters
        ----------
        x0 : np.ndarray
            new initial state

        Returns
        -------

        """
        k_offset = _calculate_index_of_closest_reference_point(x0, self.path.trajectory.x,
                                                               self.prediction_model.casadi_model['parameter']['L'])
        self.k_offset = k_offset

    def get_init_u(self):
        # assume init values are last solution for shift of init guess and development of virtual system
        self.last_z, self.last_v = self._get_initial_guess()
        self.last_loss, self.last_sol = 0, {}
        self.last_x = self.last_z[:, :self.prediction_model.nx]
        self.last_u = self.last_v[:, 0]
        self.current_virtual_state = self.last_z[1, self.prediction_model.nx:]
        return self.path.trajectory.u[0]

    def _get_new_ground_truth(self, x_new: np.ndarray):
        y = x_new - self.prediction_model.predict(self.t - self.prediction_model.Ts, self.last_x[0], self.last_u[0],
                                                  ignore_residual_model=True,
                                                  ignore_virtual_system=True)
        return y

    def _get_initial_guess(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.last_u is None:
            # get initial guess
            u_ref_init = self.path.trajectory.u[self.k_offset:(self.N + self.k_offset) % len(self.path.trajectory.u)]
            v0 = np.zeros((self.N, 1 + u_ref_init.shape[1]))
            v0[:, 0:u_ref_init.shape[1]] = self.path.trajectory.u[:self.N]

            initial_path_parameter = self.k_offset / (self.path.trajectory.x.shape[0] - 1)
            x0_virtual = np.array(
                [[self.path.path_speed * i * self.prediction_model.Ts, self.path.path_speed, 0] for i in
                 range(self.N + 1)])
            x0_virtual[:, 0] = x0_virtual[:, 0] + initial_path_parameter
            x0_system = self.path.trajectory.x[
                ((self.k_offset + np.arange(self.N + 1)) % (len(self.path.trajectory.x) - 1)).astype(int)]
            z0 = np.hstack((x0_system, x0_virtual))

            self.current_virtual_state = np.array(
                [initial_path_parameter, self.path.path_speed] + [0] * (self.prediction_model.nx_virtual - 2))

        else:
            # hot start initial guess
            v = self.last_v if not self.updated else self.output_posterior
            z = self.last_z if not self.updated else self.state_posterior
            v0 = np.vstack((v[1:], v[-1:, :]))
            z0_N = self.prediction_model.predict(self.t + (self.N - 1) * self.prediction_model.Ts, z[-1], v[-1])
            z0 = np.vstack((z[1:], z0_N.reshape(1, -1)))

        return z0, v0

    def calculate(self, x_init, delay_u):
        """ called by the simulator to calculate the control output

        Parameters
        ----------
        x_init : np.ndarray
            current state measurement
        delay_u : bool
            whether the input is applied one time step delayed

        Returns
        -------
        float
            control command u
        """

        if delay_u:
            x_init = self.prediction_model.predict(self.t, x_init, self.last_u[0], ignore_virtual_system=True)

        # do recursive update
        if self.active_algorithm is not None and self.last_sol is not None:
            self._update(x_init)

        if self.active_algorithm != 3 or self.prior_solution3 is None:
            # define initial guess values
            z0, v0 = self._get_initial_guess()

            # solve NLP
            z_init = np.concatenate((x_init, self.current_virtual_state))
            self.last_z, self.last_v, self.last_loss, self.last_sol, self.success = self.NLP.solve(self.t, z_init, z0,
                                                                                                   v0,
                                                                                                   self.cost_options)
        elif not self.updated:
            self.last_z, self.last_v, self.last_loss, self.last_sol, self.success = self.prior_solution3
        else:
            self.last_z, self.last_v, self.last_loss, self.last_sol, self.success = self.posterior_solution3

        self.last_x = self.last_z[:, :self.prediction_model.nx]
        self.last_u = self.last_v[:, 0]
        self.current_virtual_state = self.last_z[1, self.prediction_model.nx:]
        self.t += self.prediction_model.Ts
        return self.last_u[0]

    def collect_info(self):
        """ used by the simulator to collect internal information.

        Returns
        -------
        dict
            internal information containing e.g. the predicted states

        """
        info = super().collect_info()
        if self.last_sol is not None:
            info.update({
                'z_sol': self.last_z,
                'v_sol': self.last_v,
            })

        return info


class TrajectoryTrackingMPC(MPC):

    def __init__(self, Q: np.ndarray, S: Optional[np.ndarray], R: np.ndarray, final_weight: float,
                 use_soft_constraint: bool, prediction_model: PredictionModel,
                 N: int, constraints: utils.NLPConstraints, x_ref: np.ndarray, u_ref: np.ndarray,
                 discretization: str = 'direct_multiple_shooting', online_learning_options=None):
        """

        Parameters
        ----------
        Q : np.ndarray
            weighting matrix for the state penalty
        S : np.ndarray
            weighting matrix for input changes
        final_weight : float
            weighs the terminal cost
        use_soft_constraint : bool
            whether soft constrains via slack variables should be used
        prediction_model : PredictionModel
            prediction model of the controller
        N : int
            prediction horizon
        constraints : utils.NLPConstraints
            constraints of the system
        x_ref : np.ndarray
            states of the reference trajectory
        u_ref : np.ndarray
            reference inputs
        discretization : str (default='direct_multiple_shooting')
            discretization method of the optimal control problem use either
            1) 'direct_multiple_shooting' or
            2) 'direct_single_shooting'
        online_learning_options : Optional[interfaces.OnlineLearningSettings]
            defines the option used for online learning (see RGPMPC from Michael Maiworm for more details.)
            online learning will be deactivated if None
        """

        super().__init__(Q, S, final_weight, use_soft_constraint, prediction_model, N, constraints,
                         discretization, online_learning_options)

        self.R = R
        self.x_ref = x_ref
        self.u_ref = u_ref
        self.index = 0
        self.cost_options = {'x_ref': self.x_ref,
                             'u_ref': self.u_ref}

    def set_x_init(self, x0: np.ndarray):
        self.k_offset = _calculate_index_of_closest_reference_point(x0, self.x_ref,
                                                                    self.prediction_model.casadi_model['parameter'][
                                                                        'L'])

    def _stage_cost(self, x, u, k, **kwargs) -> ca.MX:
        x_error = x - kwargs['x_ref'][(k + self.index) % len(kwargs['u_ref'])]
        u_error = u - kwargs['u_ref'][(k + self.index) % len(kwargs['u_ref'])]
        return x_error.T @ self.Q @ x_error + self.R * u_error ** 2

    def _terminal_cost(self, x: ca.MX, **kwargs) -> ca.MX:
        x_error = x - kwargs['x_ref'][(self.N + self.index) % len(kwargs['u_ref'])]
        return self.final_weight * x_error.T @ self.Q @ x_error

    def reset(self):
        super().reset()
        self.index = 0

    def get_x_init(self):
        return self.x_ref[0]

    def get_init_u(self):
        self.index += 1
        self.last_u = self.u_ref[:self.N]
        self.last_x = self.x_ref[:self.N + 1]
        self.last_loss = 0
        self.last_sol = {}
        return self.u_ref[0]

    def _get_new_ground_truth(self, x_new: np.ndarray):
        y = x_new - self.prediction_model.predict(self.t - self.prediction_model.Ts, self.last_x[0], self.last_u[0],
                                                  ignore_residual_model=True)
        return y

    def _get_initial_guess(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.last_x is None:
            x0 = self.x_ref[((np.arange(self.N + 1) + self.index) % len(self.u_ref)).astype(int)]
            u0 = self.u_ref[((np.arange(self.N) + self.index) % len(self.u_ref)).astype(int)]
        else:
            x0 = self.last_x[1:, :]
            u0 = self.last_u[1:]
            u_new = u0[-1]
            u0 = np.append(u0, u_new).reshape(-1, 1)
            x_new = self.prediction_model.predict(self.t, x0[-1], u_new)
            x0 = np.vstack((x0, x_new.reshape(1, -1)))

        return x0, u0

    def calculate(self, x_init, delay_u):

        if delay_u:
            x_init = self.prediction_model.predict(self.t, x_init, self.last_u[0])

        # do recursive update
        if self.active_algorithm is not None and self.last_sol is not None:
            self._update(x_init)

        if self.active_algorithm != 3 or self.prior_solution3 is None:
            # define initial guess values
            x0, u0 = self._get_initial_guess()

            self.last_x, self.last_u, self.last_loss, self.last_sol, self.success = self.NLP.solve(self.t, x_init, x0,
                                                                                                   u0,
                                                                                                   self.cost_options)
        elif not self.updated:
            self.last_x, self.last_u, self.last_loss, self.last_sol, self.success = self.prior_solution3
        else:
            self.last_x, self.last_u, self.last_loss, self.last_sol, self.success = self.posterior_solution3

        self.last_u = self.last_u.flatten()
        self.index = (self.index + 1) % len(self.u_ref)
        self.t += self.prediction_model.Ts
        return self.last_u[0]


def _calculate_index_of_closest_reference_point(x0, x_trajectory, L):
    x0_cart = utils.transform_to_cartesian(x0, L)
    x_cart = utils.transform_to_cartesian(x_trajectory, L)
    index_closest_point = np.argmin(np.linalg.norm(x0_cart - x_cart, axis=1))
    return index_closest_point
