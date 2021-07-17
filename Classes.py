from numpy import eye, random, squeeze, round, arange, dot, matrix, linalg, array
import numpy as np
from math import pi, e
from utility_functions import covariance_cost


class Target(object):
    def __init__(self, trajectory, process_error_var, initial_model_prob, switching_prob_mat, des_covariance, dt,
                 std_measurement):
        """
        :param trajectory: [(x,y)] True location of target for each time step (list index)
        :param process_error_var: process error covariance for each model [ meter/sec^3 ]
        :param initial_model_prob: list of weights for each kalman model
        :param switching_prob_mat: models switching matrix probabilities
        :param des_covariance: desired covariance matrix for each time step ( list of des_cov)
        :param dt: time step
        :param std_measurement: std for measurement (constant for all models)
        """
        self.trajectory = trajectory
        self.P_0 = des_covariance
        self.models_weights = initial_model_prob
        self.switching_prob_mat = switching_prob_mat
        self.std_meas = std_measurement
        self.model_list = [KalmanFilter(dt, 0, q, std_measurement) for q in process_error_var]

        x_init = np.concatenate((trajectory[0][0:2], trajectory[0][2:4], trajectory[0][2:]))
        self.x = x_init
        for kf in self.model_list:
            kf.x = x_init
        self.P_pre = np.zeros_like(des_covariance)
        # for evaluation - keeping history of simulation
        self.predicted_track = []
        self.Pt = []
        self.selected_for_update = []

    def update_models(self, true_meas, option=1):
        true_meas = np.concatenate((true_meas, [0, 0]))
        noise = random.normal(0, self.std_meas, np.shape(self.model_list[0].H)[0])
        noise[2:] = noise[2:] / 100
        for model in self.model_list:
            z = dot(model.H, true_meas) + noise
            model.update(z)
        if option == 2:
            gamma = np.zeros_like(self.models_weights)
            for j, (mu, model) in enumerate(
                    zip(self.models_weights, [model for model in self.model_list])):  # updating the mixed states
                z_residual = dot(model.H, true_meas) - dot(model.H, model.x) + noise
                S = model.get_S_matrix()
                gamma[j] = ((2 * pi * linalg.det(S)) ** (-0.5)) * \
                           e ** (-0.5 * (linalg.multi_dot([z_residual.T, linalg.inv(S), z_residual])))
            temp = np.sum([mu * g for mu, g in zip(self.models_weights, gamma)])
            self.models_weights = [mu * g / temp for mu, g in zip(self.models_weights, gamma)]
            models_x_pre_mixing = [kf.x for kf in self.model_list]
            models_P_pre_mixing = [kf.P for kf in self.model_list]
            for i, model in enumerate(self.model_list):
                model.x = np.zeros_like(model.x)
                model.P = np.zeros_like(model.P)
                temp = np.sum([self.switching_prob_mat[n, i] * mu for n, mu in enumerate(self.models_weights)])
                for j, x in enumerate(models_x_pre_mixing):
                    mu_ij = self.switching_prob_mat[j, i] * self.models_weights[j] / temp
                    model.x = model.x + mu_ij * x
                for j, (P, x) in enumerate(zip(models_P_pre_mixing, models_x_pre_mixing)):
                    mu_ij = self.switching_prob_mat[j, i] * self.models_weights[j] / temp
                    model.P = model.P + mu_ij * (P + dot(model.x - x, (model.x - x).T))
            print(self.P_pre)

    def predict_models(self, update_cov=True, update_state=True):
        for model in self.model_list:
            _ = model.predict(update_cov, update_state)

    def aggregate_state(self, update=False):
        state = np.zeros_like(self.model_list[0].x)
        for mu, x in zip(self.models_weights, [model.x for model in self.model_list]):
            state = state + mu * x
        if update:  # means the target was selected for update
            self.predicted_track[-1] = state[0:2]
        else:
            self.predicted_track.append(state[0:2])
        self.x = state

    def aggregate_combined_covariance(self, save_covariance=True, update=False):
        P_pre = np.zeros_like(self.P_pre)
        for mu, model in zip(self.models_weights, [model for model in self.model_list]):
            P_pre = P_pre + mu * (model.P + dot(model.x - self.x, (model.x - self.x).T))
        self.P_pre = P_pre
        if save_covariance:
            if update:
                self.Pt[-1] = P_pre
            else:
                self.Pt.append(P_pre)
        return P_pre

    def update_model_probability_weights(self):
        weights = self.models_weights
        for j, _ in enumerate(self.model_list):
            self.models_weights[j] = np.sum(
                [self.switching_prob_mat[j, k] * mu for k, mu in enumerate(weights)])

    def get_CMR(self, g='g1'):
        return array([covariance_cost(P, self.P_0, g) / covariance_cost(self.P_0, 0, g) for P in self.Pt])


class KalmanFilter:
    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt  # time step
        self.u = u  # control input we dont use it here ( we dont have anything to control)
        self.std_acc = std_acc  # standard deviation of the acceleration

        self.A = array([[1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2, 0.0],
                        [0.0, 1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2],
                        [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        self.H = array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self.Q = array([[(dt ** 6) / 36, 0, (dt ** 5) / 12, 0, (dt ** 4) / 6, 0],
                        [0, (dt ** 6) / 36, 0, (dt ** 5) / 12, 0, (dt ** 4) / 6],
                        [(dt ** 5) / 12, 0, (dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                        [0, (dt ** 5) / 12, 0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                        [(dt ** 4) / 6, 0, (dt ** 3) / 2, 0, (dt ** 2), 0],
                        [0, (dt ** 4) / 6, 0, (dt ** 3) / 2, 0, (dt ** 2)]]) * self.std_acc
        Q_hat = array([[103.4, 0, 73.4, 0, 14.3, 0],
                       [0, 103.4, 0, 73.5, 0, 14.3],
                       [73.5, 0, 53.5, 0, 10.1, 0],
                       [0, 73.5, 0, 53.3, 0, 10.1],
                       [17.3, 0, 10.1, 0, 2, 0],
                       [0, 14.3, 0, 10.1, 0, 2]])  # as shown in the simulation section
        self.Q = np.multiply(self.Q, Q_hat)
        ra = std_meas / 300  # Noise of Acceleration Measurement
        rp = std_meas  # Noise of Position Measurement
        self.R = array([[rp, 0.0, 0.0, 0.0],
                        [0.0, rp, 0.0, 0.0],
                        [0.0, 0.0, ra, 0.0],
                        [0.0, 0.0, 0.0, ra]])
        self.P = eye(self.A.shape[1]) * 10

        self.x = array([[0], [0], [0], [0], [0], [0]])

    def predict(self, update_cov=True, update_state=True):
        # Ref :Eq.(9) and Eq.(10)

        # Update time state
        if update_state:
            # self.x = dot(self.A, self.x) + dot(self.B, self.u)
            self.x = dot(self.A, self.x)

        # Calculate error covariance
        # P= A*P*A' + Q
        if update_cov:
            self.P = dot(dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = dot(self.H, dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = dot(dot(self.P, self.H.T), linalg.inv(S))  # Eq.(11)

        self.x = self.x + dot(K, (z - dot(self.H, self.x)))  # Eq.(12)

        I = eye(self.H.shape[1])
        self.P = dot((I - (dot(K, self.H))), self.P)  # Eq.(13)

    def get_S_matrix(self):
        return np.linalg.multi_dot([self.H,self.P,self.H.T]) + self.R
