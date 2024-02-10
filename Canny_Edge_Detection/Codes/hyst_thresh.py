#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Ramic Yannick
MatrNr: 11771174
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # First step is to analyze the picture (rubens):

    row_max, col_max = edges_in.shape
    row_max = row_max - 1 # Necessary for indexing (Index starts with zero)
    col_max = col_max - 1


    '''
    Implementation of the Hysteresis Threshold
    '''

    # For each condition (Low, Weak, High) the values should be the same

    # Not necessary but to be able for a comparison I decided to create a new array

    hyst = edges_in

    t2 = high
    t1 = low

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

    bitwise_img = hyst


    ######################################################
    return bitwise_img
