#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.

Author: Ramic Yannick
MatrNr: 11771174
"""
from typing import Callable

import time

import numpy as np
import random

import sklearn.cluster

from helper_functions import *


def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Boolean array with the same length as target_points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """
    ######################################################

    # In order to avoid the primitive ransac we can do one run without the loop and use the formula from the lecture, as
    # well as the confidence to estimate the number of trials (I just copy and pasted the code above, note that the loop
    # starts after the comments appear the second time

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    # Next, I want a random generator:
    sample = np.zeros((4, 1))

    for i in range(4):
        sample[i, 0] = int(random.randint(0, len(source_points[:, 0]) - 1))

    sample = sample.astype(int)

    H_source_points = source_points[sample]
    H_target_points = target_points[sample]

    H_source_points = H_source_points.reshape(4, 2)
    H_target_points = H_target_points.reshape(4, 2)

    # After we have our sample, we want to compute now with the least square function down the Transformation Matrix H:
    # Unfortunately the code here is not working therefore I need to copy the whole process from the function down:

    ############################################IMPORTED################################################################
    # columns: 8
    n = len(H_source_points[:, 0])

    lin_system = np.zeros(shape=(2 * n, 8))
    right_side = np.zeros(shape=(2 * n, 1))

    helper_1 = 0
    helper_2 = 1

    for i in range(len(H_source_points[:, 0])):
        lin_system[helper_1, 0] = H_source_points[i, 0]
        lin_system[helper_1, 1] = H_source_points[i, 1]
        lin_system[helper_1, 2] = 1
        lin_system[helper_1, 6] = -H_target_points[i, 0] * H_source_points[i, 0]
        lin_system[helper_1, 7] = -H_target_points[i, 0] * H_source_points[i, 1]

        right_side[helper_1, 0] = H_target_points[i, 0]

        helper_1 = helper_1 + 2

        lin_system[helper_2, 3] = H_source_points[i, 0]
        lin_system[helper_2, 4] = H_source_points[i, 1]
        lin_system[helper_2, 5] = 1
        lin_system[helper_2, 6] = -H_target_points[i, 1] * H_source_points[i, 0]
        lin_system[helper_2, 7] = -H_target_points[i, 1] * H_source_points[i, 1]

        right_side[helper_2, 0] = H_target_points[i, 1]

        helper_2 = helper_2 + 2

    sol_total = np.linalg.lstsq(lin_system, right_side, rcond=-1)
    # The solution is stored in the first array of sol_total:
    sol_h = sol_total[0]

    H = np.zeros(shape=(3, 3))
    H[0, 0] = sol_h[0]
    H[0, 1] = sol_h[1]
    H[0, 2] = sol_h[2]

    H[1, 0] = sol_h[3]
    H[1, 1] = sol_h[4]
    H[1, 2] = sol_h[5]

    H[2, 0] = sol_h[6]
    H[2, 1] = sol_h[7]
    H[2, 2] = 1

    homography = H
    ####################################################################################################################

    # Next, after the homography (Matrix H) was found, I need to compute the expected outcome for all source Points, in
    # order to analyze them further:

    # The elimination of omega is given in the lecture slides, thus only two equations for one point x' and y' need to
    # be implemented for all source_points:

    x_estimation = np.zeros((len(source_points[:, 0]), 1))
    y_estimation = np.zeros((len(source_points[:, 0]), 1))

    for i in range(len(source_points[:, 0])):
        x_estimation[i, 0] = (H[0, 0] * source_points[i, 0] + H[0, 1] * source_points[i, 1] + H[0, 2]) / \
                             (H[2, 0] * source_points[i, 0] + H[2, 1] * source_points[i, 1] + H[2, 2])

        y_estimation[i, 0] = (H[1, 0] * source_points[i, 0] + H[1, 1] * source_points[i, 1] + H[1, 2]) / \
                             (H[2, 0] * source_points[i, 0] + H[2, 1] * source_points[i, 1] + H[2, 2])

    # Next, I need to define the euclidian distance which is basically the implementation of pythagoras, to define the
    # error and in conclusion the number of inliers:

    dx_squared = (target_points[:, 0] - x_estimation[:, 0]) ** 2
    dy_squared = (target_points[:, 1] - y_estimation[:, 0]) ** 2

    distance = np.sqrt(dx_squared + dy_squared)

    # For all inliers, we know the fact that the following condition must be fullfilled: distance < inlier_threshold

    idx_inlier_tuple = np.nonzero(distance < inlier_threshold)
    idx_inlier = np.array(idx_inlier_tuple)[0]

    number_in = len(idx_inlier)

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    points_total = len(source_points[:,0])


    number_trials = int(round(np.log(1-confidence)/np.log(1-((number_in/points_total)*((number_in-1)/(points_total-1))))))


    # RANSAC START HERE!!!
    primitive_ransac = number_trials

    number_inliers = np.zeros((primitive_ransac,1))
    best_H = np.zeros((3,3))
    stored_inliers = np.zeros((len(source_points[:,0]),1))

    for j in range(primitive_ransac):
        # Next, I want a random generator:
        sample = np.zeros((4,1))

        for i in range(4):
            sample[i,0] = int(random.randint(0,len(source_points[:,0])-1))

        sample = sample.astype(int)

        H_source_points = source_points[sample]
        H_target_points = target_points[sample]

        H_source_points = H_source_points.reshape(4,2)
        H_target_points = H_target_points.reshape(4,2)

        # After we have our sample, we want to compute now with the least square function down the Transformation Matrix H:
        # Unfortunately the code here is not working therefore I need to copy the whole process from the function down:

        ############################################IMPORTED################################################################
        # columns: 8
        n = len(H_source_points[:, 0])

        lin_system = np.zeros(shape=(2 * n, 8))
        right_side = np.zeros(shape=(2 * n, 1))

        helper_1 = 0
        helper_2 = 1

        for i in range(len(H_source_points[:, 0])):
            lin_system[helper_1, 0] = H_source_points[i, 0]
            lin_system[helper_1, 1] = H_source_points[i, 1]
            lin_system[helper_1, 2] = 1
            lin_system[helper_1, 6] = -H_target_points[i, 0] * H_source_points[i, 0]
            lin_system[helper_1, 7] = -H_target_points[i, 0] * H_source_points[i, 1]

            right_side[helper_1, 0] = H_target_points[i, 0]

            helper_1 = helper_1 + 2

            lin_system[helper_2, 3] = H_source_points[i, 0]
            lin_system[helper_2, 4] = H_source_points[i, 1]
            lin_system[helper_2, 5] = 1
            lin_system[helper_2, 6] = -H_target_points[i, 1] * H_source_points[i, 0]
            lin_system[helper_2, 7] = -H_target_points[i, 1] * H_source_points[i, 1]

            right_side[helper_2, 0] = H_target_points[i, 1]

            helper_2 = helper_2 + 2

        sol_total = np.linalg.lstsq(lin_system, right_side, rcond=-1)
        # The solution is stored in the first array of sol_total:
        sol_h = sol_total[0]

        H = np.zeros(shape=(3, 3))
        H[0, 0] = sol_h[0]
        H[0, 1] = sol_h[1]
        H[0, 2] = sol_h[2]

        H[1, 0] = sol_h[3]
        H[1, 1] = sol_h[4]
        H[1, 2] = sol_h[5]

        H[2, 0] = sol_h[6]
        H[2, 1] = sol_h[7]
        H[2, 2] = 1

        ####################################################################################################################

        # Next, after the homography (Matrix H) was found, I need to compute the expected outcome for all source Points, in
        # order to analyze them further:

        # The elimination of omega is given in the lecture slides, thus only two equations for one point x' and y' need to
        # be implemented for all source_points:

        x_estimation = np.zeros((len(source_points[:,0]),1))
        y_estimation = np.zeros((len(source_points[:,0]),1))

        for i in range(len(source_points[:,0])):
            x_estimation[i,0] = (H[0,0]*source_points[i,0] + H[0,1]*source_points[i,1] + H[0,2]) / \
                       (H[2,0]*source_points[i,0] + H[2,1]*source_points[i,1] + H[2,2])

            y_estimation[i,0] = (H[1,0]*source_points[i,0] + H[1,1]*source_points[i,1] + H[1,2]) / \
                       (H[2,0]*source_points[i,0] + H[2,1]*source_points[i,1] + H[2,2])

        # Next, I need to define the euclidian distance which is basically the implementation of pythagoras, to define the
        # error and in conclusion the number of inliers:

        dx_squared = (target_points[:,0] - x_estimation[:,0]) ** 2
        dy_squared = (target_points[:,1] - y_estimation[:,0]) ** 2

        distance = np.sqrt(dx_squared + dy_squared)

        # For all inliers, we know the fact that the following condition must be fullfilled: distance < inlier_threshold

        idx_inlier_tuple = np.nonzero(distance < inlier_threshold)
        idx_inlier =  np.array(idx_inlier_tuple)[0]

        number_inliers[j,0] = len(idx_inlier)

        if np.amax(number_inliers) == number_inliers[j,0]:
            best_H = H
            stored_inliers =idx_inlier
        else:
            pass


    num_iterations = number_trials
    best_suggested_homography = best_H

    best_inliers = np.full(shape=len(target_points), fill_value=False, dtype=bool)
    best_inliers[stored_inliers] = bool(True)

    '''
    best_suggested_homography = np.eye(3)
    best_inliers = np.full(shape=len(target_points), fill_value=True, dtype=bool)
    num_iterations = 0
    '''

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # Last step is to refine the projection matrix again with a least square method, with the best inliers
    # Again I just copy and pasted the code:

    n_opt = len(stored_inliers)

    lin_system_opt = np.zeros(shape=(2*n_opt, 8))
    right_side_opt = np.zeros(shape=(2*n_opt,1))

    helper_1 = 0
    helper_2 = 1

    source_inliers = source_points[stored_inliers]
    target_inliers = target_points[stored_inliers]


    for i in range(len(source_inliers[:,0])):
        lin_system_opt[helper_1,0] = source_inliers[i,0]
        lin_system_opt[helper_1,1] = source_inliers[i,1]
        lin_system_opt[helper_1,2] = 1
        lin_system_opt[helper_1,6] = -target_inliers[i,0]*source_inliers[i,0]
        lin_system_opt[helper_1,7] = -target_inliers[i,0]*source_inliers[i,1]

        right_side_opt[helper_1,0] = target_inliers[i,0]

        helper_1 = helper_1+2

        lin_system_opt[helper_2,3] = source_inliers[i,0]
        lin_system_opt[helper_2,4] = source_inliers[i,1]
        lin_system_opt[helper_2,5] = 1
        lin_system_opt[helper_2,6] = -target_inliers[i,1]*source_inliers[i,0]
        lin_system_opt[helper_2,7] = -target_inliers[i,1]*source_inliers[i,1]

        right_side_opt[helper_2,0] = target_inliers[i,1]

        helper_2 = helper_2+2

    sol_total_opt = np.linalg.lstsq(lin_system_opt,right_side_opt,rcond=-1)
    # The solution is stored in the first array of sol_total:
    sol_h_opt = sol_total_opt[0]

    H_opt = np.zeros(shape=(3,3))
    H_opt[0,0] = sol_h_opt[0]
    H_opt[0,1] = sol_h_opt[1]
    H_opt[0,2] = sol_h_opt[2]

    H_opt[1,0] = sol_h_opt[3]
    H_opt[1,1] = sol_h_opt[4]
    H_opt[1,2] = sol_h_opt[5]

    H_opt[2,0] = sol_h_opt[6]
    H_opt[2,1] = sol_h_opt[7]
    H_opt[2,2] = 1

    # Now I need to update the error (distance) as well as the number of inliers:

    x_estimation_opt = np.zeros((len(source_points[:, 0]), 1))
    y_estimation_opt = np.zeros((len(source_points[:, 0]), 1))

    for i in range(len(source_points[:, 0])):
        x_estimation_opt[i, 0] = (H[0, 0] * source_points[i, 0] + H[0, 1] * source_points[i, 1] + H[0, 2]) / \
                             (H[2, 0] * source_points[i, 0] + H[2, 1] * source_points[i, 1] + H[2, 2])

        y_estimation_opt[i, 0] = (H[1, 0] * source_points[i, 0] + H[1, 1] * source_points[i, 1] + H[1, 2]) / \
                             (H[2, 0] * source_points[i, 0] + H[2, 1] * source_points[i, 1] + H[2, 2])


    dx_squared_opt = (target_points[:, 0] - x_estimation_opt[:, 0]) ** 2
    dy_squared_opt = (target_points[:, 1] - y_estimation_opt[:, 0]) ** 2

    distance_opt = np.sqrt(dx_squared_opt + dy_squared_opt)

    idx_inlier_tuple_opt = np.nonzero(distance_opt < inlier_threshold)
    idx_inlier_opt = np.array(idx_inlier_tuple_opt)[0]

    best_suggested_homography = H_opt

    if len(idx_inlier_opt) == len(idx_inlier):
        pass
    else:
        best_inliers = np.full(shape=len(target_points), fill_value=False, dtype=bool)
        best_inliers[idx_inlier_opt] = bool(True)




    ######################################################
    return best_suggested_homography, best_inliers, num_iterations


def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################

    # Create matrix H:
    # First we need to define the Dimension of the matrix that computes the elements h:
    # for the minimum 4 Points we would have 8 equations
    # as a result for n Points (n > 4) this would lead to n * 2 equations. In conclusion the Matrix  can be described
    # with:
    #
    # rows: 2 * n
    # columns: 8
    n = len(source_points[:,0])

    lin_system = np.zeros(shape=(2*n, 8))
    right_side = np.zeros(shape=(2*n,1))

    helper_1 = 0
    helper_2 = 1


    for i in range(len(source_points[:,0])):
        lin_system[helper_1,0] = source_points[i,0]
        lin_system[helper_1,1] = source_points[i,1]
        lin_system[helper_1,2] = 1
        lin_system[helper_1,6] = -target_points[i,0]*source_points[i,0]
        lin_system[helper_1,7] = -target_points[i,0]*source_points[i,1]

        right_side[helper_1,0] = target_points[i,0]

        helper_1 = helper_1+2

        lin_system[helper_2,3] = source_points[i,0]
        lin_system[helper_2,4] = source_points[i,1]
        lin_system[helper_2,5] = 1
        lin_system[helper_2,6] = -target_points[i,1]*source_points[i,0]
        lin_system[helper_2,7] = -target_points[i,1]*source_points[i,1]

        right_side[helper_2,0] = target_points[i,1]

        helper_2 = helper_2+2

    sol_total = np.linalg.lstsq(lin_system,right_side,rcond=-1)
    # The solution is stored in the first array of sol_total:
    sol_h = sol_total[0]

    H = np.zeros(shape=(3,3))
    H[0,0] = sol_h[0]
    H[0,1] = sol_h[1]
    H[0,2] = sol_h[2]

    H[1,0] = sol_h[3]
    H[1,1] = sol_h[4]
    H[1,2] = sol_h[5]

    H[2,0] = sol_h[6]
    H[2,1] = sol_h[7]
    H[2,2] = 1


    homography = H
    #homography = np.eye(3)

    ######################################################
    return homography
