#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Yannick Ramic
MatrNr: 11771174
"""
import numpy as np
import cv2
import math

from typing import List

from helper_functions import non_max


def harris_corner(img: np.ndarray,
                  sigma1: float,
                  sigma2: float,
                  k: float,
                  threshold: float) -> List[cv2.KeyPoint]:
    """ Detect corners using the Harris corner detector

    In this function, corners in a grayscale image are detected using the Harris corner detector.
    They are returned in a list of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[cv2.KeyPoint]

    """
    
    ######################################################

    kernel_width = 2 * math.ceil(3*sigma1) + 1
    kernel = cv2.getGaussianKernel(kernel_width, sigma1)

    gauss = np.outer(kernel, kernel.transpose())

    img_blur = cv2.filter2D(img, -1, gauss)

    grad = np.gradient(img_blur)

    I_x = grad[1]
    I_y = grad[0]

    # It is better to implement the Gauss Filter here so before squaring the elements:
    kernel_width_2 = 2 * math.ceil(3 * sigma2) + 1
    kernel_2 = cv2.getGaussianKernel(kernel_width_2, sigma2)

    gauss_2 = np.outer(kernel_2, kernel_2.transpose())

    '''
    G_x = cv2.filter2D(I_x, -1, gauss_2)
    G_y = cv2.filter2D(I_y, -1, gauss_2)

    I_xy = G_x * G_y
    I_xx = G_x ** 2
    I_yy = G_y ** 2
    '''

    # Another Approach:
    I_xx = cv2.filter2D((I_x ** 2), -1, gauss_2)
    I_yy = cv2.filter2D((I_y ** 2), -1, gauss_2)
    I_xy = cv2.filter2D((I_x * I_y), -1, gauss_2)

    # So the matrix M would in the end look as followed consisting of 4 matrices itself
    M = np.array([[I_xx, I_xy],[I_xy, I_yy]])

    # Next, it is necessary to think about how to depict the determinant from M
    # My approach can be described as followed:
    # If I would have a 2x2 Matrix in the form: M = [a,b;c,d] and neglect the fact that a,b,c & d itself are matrices,
    # then the determinant of M could be easily computed as followed: det(M) = a*d - b*c
    # If the underlying matrix wouldn't be a 2x2 Matrix calculating the determinant would become a lot more difficult,
    # that is why in the next step, we treat M as a 2x2 Matrix:

    det_M = (I_xx*I_yy) - (I_xy ** 2) # Note: that this will be a Matrix again!

    # Same thoughts are applying for the trace:
    # Usually the trace for a 2x2 matrix, depicted above, follows: tr(M) = a + d
    # So also here we treat M as a 2x2 Matrix:

    tr_M = I_xx + I_yy # Note: that this will also be a Matrix again!

    # Finally we can compute now the Measure for Cornerness R as followed:

    R = det_M - k*(tr_M ** 2)

    # Now Values should be normalized so that the R_max = 1:

    norm_factor = 1/np.amax(R)
    R = norm_factor * R

    # From the lecture slides we know the following relationship:
    # 1) R < 0: Edge
    # 2) R small (R=0): Flat
    # 3) R > 0: Corner

    corner = np.where(R > 0, 1, 0)
    edges = np.where(R < 0, R, 0)

    # Next: Non Max Surpression:

    kernel_nonmax = np.ones(shape=(3, 3), dtype=np.uint8)
    kernel_nonmax[1, 1] = 0

    dilation = cv2.dilate(R, kernel_nonmax)

    # This gives back a boolean matrix with all Values True that will be stored in the array sol_nonmax:
    local_maxima = R > dilation

    sol_nonmax = np.zeros(R.shape)
    idx_nonmax = np.nonzero(local_maxima == True)
    sol_nonmax[idx_nonmax] = R[idx_nonmax]

    # Next Task is to threshold the feature values:
    hyst = sol_nonmax

    # All Values above the threshold can be set one:
    hyst = np.where(hyst >= threshold, hyst, 0)

    # Now to get the keypoints the function cv2.KeyPoint needs to be applied, therefore I need to find all x and y
    # values which represent the column and row:

    # row - y Axis
    # col - x Axis

    [row_hyst, col_hyst] = np.nonzero(hyst)

    # The x and y value need to be floats therefore I have to convert the integers into floats:
    row_hyst = row_hyst.astype(np.float32)
    col_hyst = col_hyst.astype(np.float32)
    hyst = hyst.astype(np.float32)

    # Problem that arose is that cv2.KeyPoint can only take one input therefore the only way to solve this is with a for
    # Loop. Next Problem was how to store these object, with a numpy array it wasn't possible because only boolean, int
    # and floats can be stored, therefore Lists were my choice to continue.

    keypoints = []

    for i in range(len(row_hyst)):
        keypoints.append(cv2.KeyPoint(col_hyst[i],row_hyst[i],1, 0, hyst[row_hyst[i].astype(int),col_hyst[i].astype(int)]))


    return keypoints
