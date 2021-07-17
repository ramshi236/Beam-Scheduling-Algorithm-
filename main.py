from numpy import eye, random, squeeze, round, arange, dot, matrix, linalg, array
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Classes import KalmanFilter, Target
from utility_functions import covariance_cost, generate_measurements
import string
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({'font.size': 22})


def beam_scheduling_algorithm(targets, searching=True, g='g1', criteria=2):
    names = list(string.ascii_uppercase)
    selection_acc = np.zeros((len(targets) + 1, 1))
    for k, _ in enumerate(targets[0].trajectory):
        # Update the model probability weights for each target
        # [target.update_model_probability_weights() for target in targets] # not working !
        # Predict for t=k+1 for all models in all targets( without updating their error covariance matrix)
        [target.predict_models(update_cov=False) for target in targets]
        # Aggregate the state model for each target
        [target.aggregate_state() for target in targets]
        # Get covariance for each target before covariance prediction
        cov_before_update = [target.aggregate_combined_covariance(save_covariance=False) for target in targets]
        # Update the error covariances for all targets ( complete the prediction step )
        [target.predict_models(update_state=False) for target in targets]
        [target.aggregate_combined_covariance(save_covariance=True) for target in targets]
        # Obtain the optimal control vector according to F1/F2
        F = []
        if criteria == 1:
            for j, target in enumerate(targets):
                F.append(covariance_cost(target.P_0, target.P_pre, option=g)
                         + np.sum(
                    [covariance_cost(targets[jj].P_0, cov, option=g) for jj, cov in enumerate(cov_before_update)
                     if jj is not j]))
            F.append(np.sum([covariance_cost(targets[jj].P_0, cov, g) for jj, cov in enumerate(cov_before_update)]))  # searching option
            u = np.argmax(F)
        else:  # criteria==2
            for j, target in enumerate(targets):
                cov_list = cov_before_update
                cov_list[j] = target.P_pre
                F.append(max([covariance_cost(cov,targets[jj].P_0) for jj,cov in enumerate(cov_list)]))
            F.append(max([covariance_cost(cov, targets[jj].P_0) for jj, cov in enumerate(cov_before_update)])) # searching option
            u = np.argmin(F)
        if u == len(targets) and searching:  # means that the optimal control for the radar is to search
            print("Round {} - searching".format(str(k)))
            selection_acc[0] = selection_acc[0] + 1
            continue
        else:
            u = np.argmax(F[:-1])
            selection_acc[u + 1] = selection_acc[u + 1] + 1
        # update the selected target
        print("Round {} - updating target {}".format(str(k), names[u]))
        # update the models state
        targets[u].update_models(targets[u].trajectory[k], option=1)
        # aggregate the target state and covariance
        targets[u].aggregate_state(update=True)
        targets[u].aggregate_combined_covariance(save_covariance=True, update=True)
    return selection_acc / (k + 1)


def Targets_tracking_plot(targets):
    """
    :param targets: list of Target Class
    """
    fig, axe = plt.subplots()
    beam_scheduling_algorithm(targets)
    string_targets = ['A', 'B']
    for r, target in enumerate(targets):
        axe.scatter(array(target.predicted_track)[:, 0], array(target.predicted_track)[:, 1], marker='x', s=8,
                    label="prediction - Target {}".format(string_targets[r]))
        axe.scatter(array(target.trajectory)[:, 0], array(target.trajectory)[:, 1], marker='o', s=1,
                    label="True - Target {}".format(string_targets[r]))
        axe.annotate("{} initial position".format(string_targets[r]),
                     xy=(array(target.trajectory)[0, 0:2]), xycoords='data',
                     xytext=(array(target.trajectory)[0, 0:2] - [-2000, 5000]), textcoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top')
    axe.grid()
    axe.legend()
    plt.show()


def variance_CMR_MonteCarlo_plot(dt, number_of_maneuvers, process_error_var, init_pos_list, initial_model_prob,
                                 switching_prob_mat, desired_covariances, std_meas, number_of_simulations=10):
    # generating targets with random trajectory
    number_of_targets = len(desired_covariances)
    names = list(string.ascii_uppercase)
    first_time = True
    x_var = [[] for _ in range(number_of_targets)]
    CMR = [[] for _ in range(2)]  # only for 2 targets (even though we can simulate for more)
    selection_acc_mc = np.zeros((number_of_targets + 1, 1))
    for _ in range(number_of_simulations):  # Preforming Monte-Carlo simulation
        targets = []
        for j in range(number_of_targets):
            targets.append(Target(generate_measurements(dt, number_of_maneuvers, init_pos_list[j]), process_error_var,
                                  initial_model_prob,
                                  switching_prob_mat, desired_covariances[j], dt,
                                  std_meas))
        selection_acc_mc = selection_acc_mc + (1 / number_of_simulations) * beam_scheduling_algorithm(targets)

        if first_time:
            for j, target in enumerate(targets):
                x_var[j] = array([P[0, 0] for P in target.Pt]) * (1 / number_of_simulations)
                CMR[j] = target.get_CMR() * (1 / number_of_simulations)
            first_time = False
        else:
            for j, target in enumerate(targets):
                x_var[j] = x_var[j] + array([P[0, 0] for P in target.Pt]) * (1 / number_of_simulations)
                CMR[j] = CMR[j] + target.get_CMR() * (1 / number_of_simulations)
    # print("Searching selection percent - {} ".format(str(selection_acc_mc[-1])))
    # for j in range(number_of_targets):
    #     print("Target {} selection percent - {}".format(names[j], str(selection_acc_mc[j])))
    gs = gridspec.GridSpec(2, 2)
    t = [k * dt for k in range(len(targets[0].trajectory))]
    names = list(string.ascii_uppercase)
    ax_x_var = plt.subplot(gs[0, :])
    for j, _ in enumerate(targets):
        ax_x_var.plot(t, x_var[j], label="Target - {}".format(names[j]))
    ax_x_var.plot(t, [targets[0].P_0[0, 0] for _ in range(len(t))], linestyle='--', label="Desired x variance - A")
    ax_x_var.plot(t, [targets[1].P_0[0, 0] for _ in range(len(t))], linestyle='--', label="Desired x variance - B")
    ax_CMR_A = plt.subplot(gs[1, 0])
    ax_CMR_A.plot(t, CMR[0], label="Target - {}".format(names[0]))
    ax_CMR_B = plt.subplot(gs[1, 1])
    ax_CMR_B.plot(t, CMR[1], label="Target - {}".format(names[1]))
    ax_x_var.grid()
    ax_CMR_A.grid()
    ax_CMR_B.grid()
    ax_x_var.legend()
    ax_CMR_A.legend()
    ax_CMR_B.legend()
    plt.show()


if __name__ == '__main__':
    initial_model_prob = array([0.5, 0.4, 0.1])
    process_error_var = array([0.1, 10, 100])
    switching_prob_mat = array([[0.95, 0.025, 0.025],
                                [0.025, 0.95, 0.025],
                                [0.05, 0.05, 0.9]])
    des_cov1 = np.diag([250, 250, 25, 25, 1.5, 1.5])
    des_cov2 = np.diag([250, 250, 25, 25, 1.5, 1.5])
    desired_covariances = [des_cov1, des_cov2]
    std_meas = 30
    dt = 0.2
    number_of_maneuvers = 5

    A_target = Target(generate_measurements(dt, number_of_maneuvers, [1000, 10000]), process_error_var,
                      initial_model_prob,
                      switching_prob_mat, des_cov1, dt,
                      std_meas)
    B_target = Target(generate_measurements(dt, number_of_maneuvers, [-30000, -3000]), process_error_var,
                      initial_model_prob,
                      switching_prob_mat, des_cov2, dt,
                      std_meas)
    targets = [A_target, B_target]

    Targets_tracking_plot(targets)

    # Preforming Monte-Carlo simulation
    number_of_maneuvers = 5
    number_of_simulations = 100
    init_pos_list = [[10000, 10000], [-30000, -30000]]
    variance_CMR_MonteCarlo_plot(dt, number_of_maneuvers, process_error_var, init_pos_list, initial_model_prob,
                                 switching_prob_mat, desired_covariances, std_meas, number_of_simulations)
