#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Descriptor function

Author: Yannick Ramic
MatrNr: 11771174
"""

from typing import List, Tuple

import numpy as np
import cv2

def compute_descriptors(img: np.ndarray,
                        keypoints: List[cv2.KeyPoint],
                        patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """ Calculate a descriptor on patches of the image, centred on the locations of the KeyPoints.

    Calculate a descriptor vector for each keypoint in the list. KeyPoints that are too close to the border to include
    the whole patch are filtered out. The descriptors are returned as a k x m matrix with k being the number of filtered
    KeyPoints and m being the length of a descriptor vector (patch_size**2). The descriptor at row i of
    the descriptors array is the descriptor for the KeyPoint filtered_keypoint[i].

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param keypoints: Locations at which to compute the descriptors (n x 2)
    :type keypoints: List[cv2.KeyPoint]

    :param patch_size: Value defining the width and height of the patch around each keypoint to calculate descriptor.
    :type patch_size: int

    :return: (filtered_keypoints, descriptors):
        filtered_keypoints: List of the filtered keypoints.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner.
            with k being the number of descriptors and m being the length of a descriptor (usually patch_size**2).
            The descriptor at row i belongs to the KeyPoint at filtered_keypoints[i]
    :rtype: (List[cv2.KeyPoint], np.ndarray)
    """
    ######################################################
    keypoints_total = keypoints

    # First step is to extract all x and y components from each keypoint, this can easily be achieved as followed:
    row = np.zeros(len(keypoints_total))
    col = np.zeros(len(keypoints_total))

    for i in range(len(keypoints_total)):
        col[i] = keypoints_total[i].pt[0]
        row[i] = keypoints_total[i].pt[1]

    # Next to analyse potential ghost points inside a patch I need to define the boundaries of the underlying image:
    image_shape = img.shape
    row_max = image_shape[0]-1
    col_max = image_shape[1]-1
    col_min = row_min = 0

    # For the patch size we have to consider an even number as well as an odd number:

    modulo_check = (patch_size % 2)

    if modulo_check == 1:

        # if number is odd

        patch_size_odd = patch_size  # Here the Patch Size is an odd number

        # First step is to analyze the upper boundary if there are potential ghost points:
        row_logic_o = np.where((row - ((patch_size_odd - 1) / 2)) >= row_min, row, False)
        idx_filter_1 = np.nonzero(row_logic_o > 0)
        row_adapted_1 = row[idx_filter_1]
        col_adapted_1 = col[idx_filter_1]

        # Next, I will check whether ghost points exist at the lower boundary at row_max:
        row_logic_low = np.where((row_adapted_1 + ((patch_size_odd - 1) / 2)) > row_max, False, row_adapted_1)
        idx_filter_2 = np.nonzero(row_logic_low > 0)
        row_adapted_2 = row_adapted_1[idx_filter_2]
        col_adapted_2 = col_adapted_1[idx_filter_2]

        # Same process for the left Boundary at col = 0 (col_min):
        col_logic_l = np.where((col_adapted_2 - ((patch_size_odd - 1) / 2)) >= col_min, col_adapted_2, False)
        idx_filter_3 = np.nonzero(col_logic_l > 0)
        row_adapted_3 = row_adapted_2[idx_filter_3]
        col_adapted_3 = col_adapted_2[idx_filter_3]

        # Last process at the right boundary at col_max:
        col_logic_r = np.where((col_adapted_3 + ((patch_size_odd - 1) / 2)) > col_max, False, col_adapted_3)
        idx_filter_4 = np.nonzero(col_logic_r > 0)
        row_adapted_4 = row_adapted_3[idx_filter_4]
        col_adapted_4 = col_adapted_3[idx_filter_4]
    else:
        # Case for even Number:
        patch_size_even = patch_size
        # Now I have to choose arbitrary a close point of center:
        # Chosen point: right upper point from the center

        # First step is to analyze the upper boundary if there are potential ghost points:
        row_logic_o = np.where((row-((patch_size_even-2)/2)) >= row_min, row, False)
        idx_filter_1 = np.nonzero(row_logic_o > 0)
        row_adapted_1 = row[idx_filter_1]
        col_adapted_1 = col[idx_filter_1]

        # Next, I will check whether ghost points exist at the lower boundary at row_max:
        row_logic_low = np.where((row_adapted_1 + (patch_size_even/2)) > row_max, False, row_adapted_1)
        idx_filter_2 = np.nonzero(row_logic_low > 0)
        row_adapted_2 = row_adapted_1[idx_filter_2]
        col_adapted_2 = col_adapted_1[idx_filter_2]

        # Same process for the left Boundary at col = 0 (col_min):
        col_logic_l = np.where((col_adapted_2 - (patch_size_even/2)) >= col_min, col_adapted_2, False)
        idx_filter_3 = np.nonzero(col_logic_l > 0)
        row_adapted_3 = row_adapted_2[idx_filter_3]
        col_adapted_3 = col_adapted_2[idx_filter_3]

        # Last process at the right boundary at col_max:
        col_logic_r = np.where((col_adapted_3 + ((patch_size_even-2)/2)) > col_max, False, col_adapted_3)
        idx_filter_4 = np.nonzero(col_logic_r > 0)
        row_adapted_4 = row_adapted_3[idx_filter_4]
        col_adapted_4 = col_adapted_3[idx_filter_4]


    # Summary:
    row_filt = row_adapted_4.astype(np.float32)
    col_filt = col_adapted_4.astype(np.float32)

    # To create Keypoints again I copy and paste the function from harris_corner.py

    keypoints_filt = []

    for i in range(len(row_filt)):
        keypoints_filt.append(cv2.KeyPoint(col_filt[i],row_filt[i],1, 0, img[row_filt[i].astype(int),col_filt[i].astype(int)]))

    # Resulting keypoints are as followed:
    filtered_keypoints = keypoints_filt

    # Now I need describe the descriptors
    descriptors = np.zeros(shape=(len(keypoints_filt), patch_size ** 2))

    # adapt row and columns again to integers:
    row_f = row_filt.astype(int)
    col_f = col_filt.astype(int)


    # Again Decision whether odd or even:
    if (patch_size % 2) == 0:
        patch_size_even = patch_size

        # For each Boundary:
        # u ... upper Boundary
        # lower ... lower Boundary
        # l ... left Boundary
        # r ... right Boundary

        dx_u = (patch_size_even - 2) / 2
        dx_lower = patch_size / 2
        dx_l = patch_size / 2
        dx_r = (patch_size - 2) / 2

        # Creating a Kernel with the patch_size_odd:

        # Now Loop to fill the descriptor
        for i in range(len(keypoints_filt)):
            kernel_even = img[int(row_f[i]-dx_u):int(row_f[i]+dx_lower+1),int(col_f[i]-dx_l):int(col_f[i]+dx_r+1)]
            reshaped_even = np.reshape(kernel_even,patch_size ** 2)
            descriptors[i,:] = reshaped_even
    else:
        patch_size_odd = patch_size
        dx = (patch_size_odd-1)/2

        # Creating a Kernel with the patch_size_odd:

        # Now Loop to fill the descriptor
        for i in range(len(keypoints_filt)):
            kernel_odd = img[int(row_f[i]-dx):int(row_f[i]+dx+1),int(col_f[i]-dx):int(col_f[i]+dx+1)]
            reshaped_odd = np.reshape(kernel_odd,patch_size ** 2)
            descriptors[i,:] = reshaped_odd


    return filtered_keypoints, descriptors
