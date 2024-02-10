#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Ramic Yannick
MatrNr: 11771174
"""

import cv2
import numpy as np
import math


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    # Write your own code here


    # First step is to get the orientations in degree instead of radiants, this makes life easier:

    orientations = orientations.copy() * (180/math.pi)

    # Turn all Values into positive angles:

    orientations = np.where(orientations >= 0, orientations, 360 + orientations)

    # Next step is to set all values 360 Degree to zero otherwise the normalization would lead to an error for those
    # values:

    orientations = np.where(orientations == 360, 0, orientations)

    # Now the next step is to divide the degree only in 0, 45, 90, 135, 180, 225, 270, 315 and in the End 360 = 0
    # This can easily be done with the np.where function:

    orientations = np.where((0 <= orientations) & (orientations <= 22.5), 0, orientations)
    orientations = np.where((337.5 < orientations) & (orientations <= 360), 0, orientations)
    orientations = np.where((22.5 < orientations) & (orientations <= 67.5), 45, orientations)
    orientations = np.where((67.5 < orientations) & (orientations <= 112.5), 90, orientations)
    orientations = np.where((112.5 < orientations) & (orientations <= 157.5), 135, orientations)
    orientations = np.where((157.5 < orientations) & (orientations <= 202.5), 180, orientations)
    orientations = np.where((202.5 < orientations) & (orientations <= 247.5), 225, orientations)
    orientations = np.where((247.5 < orientations) & (orientations <= 292.5), 270, orientations)
    orientations = np.where((292.5 < orientations) & (orientations <= 337.5), 315, orientations)

    # Now the array has 8 possible conditions and we want to reduce it to 4 because we always take both neighbours:
    # 0 <-> 180
    # 45 <-> 225
    # 90 <-> 270
    # 135 <-> 315

    orientations = np.where(orientations == 180, 0, orientations)
    orientations = np.where(orientations == 225, 45, orientations)
    orientations = np.where(orientations == 270, 90, orientations)
    orientations = np.where(orientations == 315, 135, orientations)

    '''
    # There is an easy way to find out if the orientation vector is correct and whether there are only elements in the 
    # array with the values 0, 45, 90, 135 Degree:
    
    # To make the code work, delete the comment lines
    
    test = [0, 45, 90, 135] # A list is completely fine for this operation instead of an np.array
    print('If the Number zero appears everything worked fine: ', np.count_nonzero(np.isin(orientations, test) == False)
    '''

    # Create an array with zeros and the shape of the orientations and gradients array:

    sol = np.zeros(gradients.shape) # I want to store all values in here this should be in the end our array edges

    row_max, col_max = gradients.shape

    # row_0 and col_0 delivers all the indices where the orientations is zero. This is necessary in order to check first
    # all values of the gradient where the orientation is 0 or 180 degree. Therefore I have the check the neighbours
    # left and right of the gradient(row_0, col_0)

    '''
    Calculations for Orientation = 0 & 180 Degree:
    '''

    r_0, c_0 = np.nonzero(orientations == 0)

    # Next I need to check if col_0 + 1 or col_0 -1 is inside the picture and doesn't represent a ghost point. Due to
    # the fact that np.max(c_0) = 575 and np.min(c_0) = 1 both operations c_0 - 1 and c_0 + 1 are possible.

    helper_0_0 = np.where(gradients[r_0, c_0] >= gradients[r_0, (np.where(c_0 == (col_max-1), (col_max-2), c_0) + 1)], True, False) # Before: (c_0 + 1)
    helper_0_1 = np.where(gradients[r_0, c_0] >= gradients[r_0, (np.where(c_0 == 0, 1, c_0) - 1)], True, False) # Before: (c_0 - 1)


    logic_0 = np.where((helper_0_0 * helper_0_1) == 1, True, False)
    idx_true_0 = np.nonzero(logic_0 == True)

    row_0 = r_0[idx_true_0]
    col_0 = c_0[idx_true_0]

    sol[row_0, col_0] = gradients[row_0, col_0]

    '''
    Calculations for Orientation = 0 & 180 Degree:
    '''


    r_90, c_90 = np.nonzero(orientations == 90)

    # Next I need to check if row_90 + 1 or row_0 -1 is inside the picture and doesn't represent a ghost point. In this
    # Case, there are ghost points. Therfore I need to set the Values of row at the boarder for the row_90 +1
    # Calculations one value lower. Due to the fact that I have in the np.where an >= sign this will give me always a
    # True Value if the indices is outside the matrix

    helper_90_0 = np.where(gradients[r_90, c_90] >= gradients[(np.where(r_90 == (row_max-1), (row_max-2), r_90) + 1),
                                                              c_90], True, False)
    helper_90_1 = np.where(gradients[r_90, c_90] >= gradients[(np.where(r_90 == 0, 1, r_90) - 1), c_90], True, False)

    logic_90 = np.where((helper_90_0 * helper_90_1) == 1, True, False)
    idx_true_90 = np.nonzero(logic_90 == True)

    row_90 = r_90[idx_true_90]
    col_90 = c_90[idx_true_90]

    sol[row_90, col_90] = gradients[row_90, col_90]

    '''
    Calculations for 45 and 225 Degrees:
    '''
    r_45, c_45 = np.nonzero(orientations == 45)

    # col_max = 577 and np.max(c_45) = 575 ---> np.max(c_45) < 576 This means that with c_45 + 1 there is no ghost point
    # col_min = 0 and np.min(c_45) = 1 ---> np.min(c_45) > 0 This means that with c_45 - 1 there is no ghost point
    # row_max = 974 and np.max(r_45) = 972 ---> np.max(r_45) < 973 This means that with r_45 + 1 there is no ghost point
    # row_min = 0 and np.min(r_45) = 1 ---> np.min(r_45) > 0 This means that with r_45 - 1 there is no ghost point

    # Comment: This is just for this picture if there is a ghost point, I should change the code similar to the code
    # with an 90 degree angle orientation.

    helper_45_0 = np.where(gradients[r_45, c_45] >= gradients[(r_45 + 1), (c_45 + 1)], True, False)
    helper_45_1 = np.where(gradients[r_45, c_45] >= gradients[(r_45 - 1), (c_45 - 1)], True, False)

    logic_45 = np.where((helper_45_0 * helper_45_1) == 1, True, False)
    idx_true_45 = np.nonzero(logic_45 == True)

    row_45 = r_45[idx_true_45]
    col_45 = c_45[idx_true_45]

    sol[row_45, col_45] = gradients[row_45, col_45]

    '''
    Calculations for 135 and 315 Degrees:
    '''

    r_135, c_135 = np.nonzero(orientations == 135)

    # Again after checking np.max and np.min of the arrays r_135 and c_135 there is no further problem with ghost points
    # As a result there is no need to do some extra work

    # helper_135_0 researches the values of the neighbour in 315 Degree angle
    # helper_135_1 researches the values of the neighbour in 135 Degree angle

    helper_135_0 = np.where(gradients[r_135, c_135] >= gradients[(r_135 - 1), (c_135 + 1)], True, False)
    helper_135_1 = np.where(gradients[r_135, c_135] >= gradients[(r_135 + 1), (c_135 - 1)], True, False)

    logic_135 = np.where((helper_135_0 * helper_135_1) == 1, True, False)
    idx_true_135 = np.nonzero(logic_135 == True)

    row_135 = r_135[idx_true_135]
    col_135 = c_135[idx_true_135]

    sol[row_135, col_135] = gradients[row_135, col_135]

    edges = sol.astype('float32')

    # Comment:
    # np.max(edges) = 0.7176146 <= 1 and np.min(edges) = 0 >= 0, Both criteria are fullfilled!
    # edges.dtype = float 32, Type is correct



    ######################################################

    return edges
