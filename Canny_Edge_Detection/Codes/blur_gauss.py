#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Yannick Ramic
MatrNr: 11771174
"""
import math

import cv2
import numpy as np

def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################

    kernel_width = 2 * math.ceil(3*sigma) + 1


    # Shape of the kernel (n x n) Matrix
    kernel = np.ones((kernel_width,kernel_width))

    # Next: x and y need to be defined... the kernel width gives back an odd number at all times
    # Result: x_max = y_max = (kernel_width-1)/2 ... This is because the center of the matrix is set to zero
    # Result: x_min = y_min = - x_max
    # As a result it is necessary to build an array for x and y with every possible number indices

    idx_min = -(kernel_width-1)/2
    idx_max = -idx_min

    x = np.linspace(idx_min,idx_max,kernel_width).astype(int)
    y = x

    # Due to the fact that after choosing sigma the kernel has always the same size, it can be computed with loops,
    # without loosing too much computational time:

    idx_i = np.linspace(0, kernel_width-1, kernel_width).astype(int)
    idx_j = idx_i

    for i in idx_i:
        for j in idx_j:
            kernel[i,j] = (1/(2*math.pi*sigma**2))*math.exp(-(x[i]**2 + y[j]**2)/(2*sigma**2))

    '''
    Comment: 
    The maximum should be in the middle of the matrix, whereas all the edges of the Matrix should have the same value.
    I checked the Matrix by debugging and saw that the result fits our expectations!
    '''

    img_blur = cv2.filter2D(img,-1,kernel)


    ######################################################
    return img_blur
