import numpy as np
from numpy import random, arange, array
from scipy import interpolate


def generate_measurements(dt, number_of_maneuvers, initial_pos, length_of_segment=100):
    flag = True
    while flag:
        try:
            t = arange(0, length_of_segment, dt)
            first_time = True
            measurement = []
            for n in range(number_of_maneuvers):
                # variance is approx (b-a)**2/12
                a_x = random.uniform(-5, 5)  # var = 8
                v_x = random.uniform(-25, 25)  # var = 208
                a_y = random.uniform(-5, 5)
                v_y = random.uniform(-25, 25)
                if first_time:
                    traj = array(
                        [[a_x * k ** 2 + v_x * k + initial_pos[0], a_y * k ** 2 + v_y * k + initial_pos[1]] for k in t])
                    measurement = array([np.concatenate((traj[n], [a_x, a_y])) for n in range(len(traj))]).T
                    first_time = False
                else:
                    traj = array(
                        [measurement.T[-1][0:2] + [a_x * k ** 2 + v_x * k, a_y * k ** 2 + v_y * k] for k in t]).T
                    traj = array([np.concatenate((traj[:, n], [a_x, a_y])) for n in range(traj.shape[1])]).T
                    measurement = np.concatenate((measurement, traj), axis=1)
            tck, u = interpolate.splprep(measurement, s=50000)
            if len(tck[0]) > 45:  # maximum number of knots points
                continue
        except:
            ValueError
            continue
        else:
            flag = False
    new_points = array(interpolate.splev(u, tck)).T
    return new_points


def covariance_cost(P_0, P,option = 'g1'):
    if option=='g1':
        return np.sum(np.trace([np.abs(P - P_0)]))
    else:
        return np.linalg.det(np.abs(P-P_0))

