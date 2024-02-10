#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Yannick Ramic
MatrNr: 11771174
"""

import cv2
import numpy as np
import math


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################
    # Write your own code here



    # First step is to formulate the Kernel in x and y direction

    kernel_x = np.ones((3,3))
    kernel_y = kernel_x

    kernel_x[0,0] = kernel_x[2,0] = -1
    kernel_x[1,0] = -2

    for i in np.linspace(0,2,3).astype(int):
        kernel_x[i,1] = 0
        kernel_x[i,2] = -kernel_x[i,0]

    kernel_y = kernel_x.transpose()

    G_x = cv2.filter2D(img, -1, kernel_x)
    G_y = cv2.filter2D(img, -1, kernel_y)

    '''
    Probably there is an easier method but with breakpoints it is easily to check whether the data type is float32
    Set Breakpoint here:
    print(type(G_x[1,1])
    print(type(G_y[1,1])
    '''

    # Operations with Matrices (arrays) are relatively simple, therefore the gradient can be computed as followed:

    gradient = np.sqrt(G_x**2 + G_y**2)

    '''
    This was my first approach to solve the problem before I have recognised that the function np.arctan2 solves the 
    problem.
    
    Though those three steps are not necessary they helped to understand the data in the picture better

    G_x_zero = G_x == 0 # Zeros in x Direction (Boolean Values)
    G_y_zero = G_y == 0 # Zeros in y Direction (Boolean Values)
    G_overlap = G_x_zero * G_y_zero # Find values where x & y are zero

    '''

    # One problem with the sobel filter is that the max and min values are over one therefore it is necessary to
    # normalize the picture in the end

    if np.max(gradient) > 1:
        max_val = np.max(gradient)
    else:
        max_val = 1

    # Due to the fact that the values of the gradient are always positive the min value doesn't play a role here

    # Next step is to compute the orientation but to understand the data better it is necessary to get a result in
    # Degree and not in radiants:

    orientation = np.arctan2(G_y,G_x) # * (180/math.pi) # This step would be necessary to have the result in degrees

    # The max value now of the orientation array is 180 degree, whereas the min value is -180 degree
    # There is an easy way to solve that issue by applying the np.where function

    '''
    
    orientation = np.where(orientation>=0, orientation, 360+orientation)

    # Next step is to set all values 360 Degree to zero otherwise the normalization would lead to an error for those
    # values:

    orientation = np.where(orientation!=360, orientation, 0)

    # Normalize Both values:
    orientation = orientation /360
    gradient = gradient / max_val
    
    '''



    ######################################################
    return gradient, orientation
