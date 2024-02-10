#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Ramic Yannick
MatrNr: 11771174
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of pixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################


    '''
    Find a Proper t1 and t2:
    '''
    # t2 represents the high value
    # t1 represents the low value



    n_row, n_col = edges_in.shape
    n_total = n_row * n_col

    vec_total = np.reshape(edges_in, n_total)

    # Due to the fact that I want just the edge pixels I have to delete every zero:
    idx_0 = np.nonzero(vec_total == 0)
    vec_total = np.delete(vec_total, idx_0)


    values, counts = np.unique(vec_total, return_counts=True)

    sum = np.dot(values,counts)

    # find t1:

    val_t1 = 1 - low_prop
    val_1 = 0
    i = 0

    while val_1 <= val_t1:
        val_1 = val_1 + ((values[i]*counts[i])/sum)
        i = i + 1

    t1 = values[i]

    # find t2:
    val_t2 = 1 - high_prop
    val_2 = 0
    j = 0

    while val_2 <= val_t2:
        val_2 = val_2 + ((values[j]*counts[j])/sum)
        j = j + 1

    t2 = values[j]




    '''
    Implementation of the Hysteresis Threshold
    '''

    # For each condition (Low, Weak, High) the values should be the same

    # Not necessary but to be able for a comparison I decided to create a new array

    hyst = edges_in

    i_col, i_row = np.where(hyst >= t2)

    # High Threshold:
    # All Values can be set to 1:
    hyst = np.where(hyst >= t2, 1, hyst)

    # Low Threshold:
    # All Values can be set to 0:
    if t1 > 0:
        hyst = np.where(hyst < t1, 0, hyst)

    # Weak Threshold (Between High and Low):
    # All Values will be set to 0.5, because it is not clear yet whether they are strong (1) or low (0)
    hyst = np.where((hyst >= t1) & (hyst < t2), 0.5, hyst)


    hysteresis = hyst * 255
    hysteresis = np.uint8(hysteresis)

    components = cv2.connectedComponents(hysteresis, 8, cv2.CV_32S)

    n_comp, tupel_comp = components

    for i in range(1, (n_comp-1)):
        row, col = np.where(tupel_comp == i)
        logic = np.isin(hyst[row, col],1)
        if np.count_nonzero(logic == True) > 0:
            hyst[row, col] = 1
        elif np.count_nonzero(logic == True) == 0:
            hyst[row, col] = 0

    hyst_out = hyst


    ######################################################
    return hyst_out
