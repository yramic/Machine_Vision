#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find clusters of pointcloud
Author: Ramic Yannick
"""

from typing import Tuple
import scipy #added
import numpy as np
import open3d as o3d
from scipy.spatial import distance
from scipy.stats import anderson
import matplotlib.pyplot as plt
from helper_functions import plot_clustering_results, silhouette_score
from helper_functions import * #ADDED just for debugging reasons
from pathlib import Path #ADDED just for debugging reasons
import time

def kmeans(points: np.ndarray,
           n_clusters: int,
           n_iterations: int,
           max_singlerun_iterations: int,
           centers_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the k-means algorithm.

    :param points: The (down-sampled) points of the pointcloud to be clustered.
    :type points: np.ndarray with shape=(n_points, 3)

    :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
    :type n_clusters: int

    :param n_iterations: Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :param centers_in: Start centers of the k-means algorithm.  If centers_in = None, the centers are randomly sampled
        from input data for each iteration.
    :type centers_in: np.ndarray with shape = (n_clusters, 3) or None

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################

    #Every 3 rows of store_center_total contain one iteration
    #Every row of store_label_total contains the label for each iteration

    store_centers_total = np.zeros((n_clusters,n_iterations*3))
    store_label_total = np.zeros((np.shape(points)[0],n_iterations))
    inertia = np.zeros((1, n_iterations)) # Also I need the inertia to evaluate at the end


    iteration_process = 0
    while iteration_process < n_iterations:

        points_shape = np.shape(points)

        if centers_in is None:
            random_indices_row = np.random.choice(range(points_shape[0]), n_clusters, replace=False)
            center_0_k = points[random_indices_row,:]

        else:
            center_0_k = centers_in

        # Initial values for the loop
        center_it = center_0_k # Relevant for the first iteration:
        center_it_new = center_it
        center_it_old = np.zeros(center_it.shape)

        iterations_single = 0

        while iterations_single < max_singlerun_iterations and (center_it_new == center_it_old).all() == False:

            center_it = center_it_new
            center_it_old = center_it

            distance_it = distance.cdist(points, center_it)
            idx_distance = np.argmin(distance_it, axis=1)

            # This was my first approach but due to the number of loops the code was way to slow, so I had to make some
            # changes
            ############################################################################################################
            #distance_helper = np.zeros((n_clusters, 1))
            #distance = np.zeros((points_shape[0], 1))

            #for i in range(points_shape[0]):
                #for j in range(n_clusters):
                    #distance_helper[j,0] = np.sqrt((points[i,0]-center_it[j,0])**2 + (points[i,1]-center_it[j,1])**2 + \
                                                   #(points[i,2]-center_it[j,2])**2)
                    # Find idx with mean value, but note idx is here an array
                    #min_val = np.min(distance_helper)
                    #idx = np.nonzero(distance_helper == min_val)
                    #idx = idx[0][0]
                    #min_val, idx = min([(abs(val), idx) for (idx, val) in enumerate(distance_helper)])

                    #distance[i,0] = idx
        ################################################################################################################

            # Now i need again a loop for the number of cluster points again in order to have a new center:
            center_it = np.zeros(center_0_k.shape) # New inertia points for each cluster should be stored here!

            for k in range(n_clusters):
                row = np.nonzero(idx_distance == k)
                # Check if array is not empty, otherwise python runs into problems
                if len(row[0])>0:
                    mean_val_x = np.mean(points[row,0])
                    mean_val_y = np.mean(points[row,1])
                    mean_val_z = np.mean(points[row,2])

                    center_it[k,0] = mean_val_x
                    center_it[k,1] = mean_val_y
                    center_it[k,2] = mean_val_z



            center_it_new = center_it
            iterations_single += 1

        store_centers_total[:,(iteration_process*3):(iteration_process*3 + 3)] = center_it_new[:,:]
        store_label_total[:,iteration_process] = idx_distance[:]

        # Also the inertia for each loop needs to be calculated: That can be achieved here, since we are already
        # in a loop
        distance_helper_i = np.zeros((points_shape[0], 1))

        distance_i = idx_distance.astype(int) #This array gives the information with which cluster the point needs to be compared

        for idx_points in range(points_shape[0]):
            distance_helper_i[idx_points,0] = ((points[idx_points,0] - center_it[distance_i[idx_points],0]) ** 2 + \
                                               (points[idx_points,1] - center_it[distance_i[idx_points],1]) ** 2 + \
                                               (points[idx_points,2] - center_it[distance_i[idx_points],2]) ** 2)

        sum = np.sum(distance_helper_i)

        inertia[0,iteration_process] = sum

        iteration_process += 1


    # Evaluate minimal inertia to know which iteration delivered the best results:

    min_inertia = np.min(inertia)
    not_rel, opt_iteration_list = np.nonzero(inertia == min_inertia)

    opt_iteration = opt_iteration_list[0]

    opt_centers = store_centers_total[:,(opt_iteration*3):(opt_iteration*3 + 3)]
    centers = opt_centers

    opt_labels = store_label_total[:,opt_iteration]

    opt_labels = opt_labels.astype(int)
    labels = opt_labels.reshape((1,len(points)))
    labels = labels[0]

    return centers, labels


def iterative_kmeans(points: np.ndarray,
                     max_n_clusters: int,
                     n_iterations: int,
                     max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Applies the k-means algorithm multiple times and returns the best result in terms of silhouette score.

    This algorithm runs the k-means algorithm for all number of clusters until max_n_clusters. The silhouette score is
    calculated for each solution. The clusters with the highest silhouette score are returned.

    :param points: The (down-sampled) points of the pointcloud that should be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_n_clusters: The maximum number of clusters that is tested.
    :type max_n_clusters: int

    :param n_iterations: Number of time each k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_iterations consecutive runs in terms of inertia.
    :type n_iterations: int

    :param max_singlerun_iterations: Maximum number of iterations of each k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points),) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    # Create a loop over mutlitple number of clusters (Note: number of clusters = 1 to max_n_clusters)

    # Each value needs to be stored:
    n_points = points.shape[0]
    centers_total = np.zeros((max_n_clusters,3*max_n_clusters))
    labels_total = np.zeros((n_points,max_n_clusters))
    score_it = np.zeros(max_n_clusters)


    for i in range(1,max_n_clusters):
        centers_it, labels_it = kmeans(points,i,n_iterations,max_singlerun_iterations,centers_in=None)
        centers_total[0:i,(3 * (i-1)):(3 * (i-1) + 3)] = centers_it
        labels_total[:,(i-1)] = labels_it
        score_it[i-1] = silhouette_score(points, centers_it, labels_it)

    # Now find highest score:
    idx_highest = np.nonzero(score_it == np.max(score_it))
    idx = idx_highest[0][0]
    #idx = idx-1 #NOTE!!!!!!!!!!!!!!!!!!!!!!!

    centers = centers_total[0:(idx+1),(3*idx):((3*idx)+3)]
    labels = labels_total[:,idx]
    labels = labels.astype(int)

    #centers = np.zeros(shape=(max_n_clusters, 3), dtype=np.float32)
    #labels = np.zeros(shape=(len(points),), dtype=int)

    return centers, labels


def gmeans(points: np.ndarray,
           tolerance: float,
           max_singlerun_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Find clusters in the provided data coming from a pointcloud using the g-means algorithm.

    The algorithm was proposed by Hamerly, Greg, and Charles Elkan. "Learning the k in k-means." Advances in neural
    information processing systems 16 (2003).

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param max_singlerun_iterations: Maximum number of iterations of the k-means algorithm for a single run.
    :type max_singlerun_iterations: int

    :return: (best_centers, best_labels)
        best_centers: Array with the centers of the calculated clusters (shape = (n_clusters, 3) and dtype=np.float32)
        best_labels: Array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label i corresponds with the center in best_centers[i] and therefore are in the range [0, n_clusters-1]
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    ######################################################
    k = 1

    chosen_centers_old = None
    chosen_centers = np.array([0,0,0])

    old_shape = 0
    new_shape = 1


    #(chosen_centers == chosen_centers_old).all()==False

    while old_shape != new_shape:


        if k == 1:
            center_0 = np.mean(points,0) # Takes the mean from the x y and z value
            chosen_centers_old = center_0
            centers_old_tot = center_0
            labels_new_tot = np.zeros(len(points[:,0]))
            old_shape = 1

        else:
            center_0 = chosen_centers
            chosen_centers_old = center_0
            centers_old_tot, labels_new_tot = kmeans(points, k, 1, max_singlerun_iterations, chosen_centers_old)
            center_0 = centers_old_tot
            old_shape = len(chosen_centers[:, 0])


        for i in range(k):

            if k == 1:
                center_loop = center_0
            else:
                center_loop = center_0[i,:]

            idx_labels_loop = np.nonzero(labels_new_tot == i)
            idx_labels_loop = idx_labels_loop[0]
            points_loop = points[idx_labels_loop]


            if points_loop.shape[0] > 1:

                cov_it = np.cov(np.transpose(points_loop))
                eigenvalue, eigenvector = np.linalg.eig(cov_it)

                highest_eig_value = np.max(eigenvalue)
                idx_best = np.nonzero(eigenvalue == highest_eig_value)
                eig_vec = eigenvector[:,idx_best]

                helper_eig = eig_vec*np.sqrt(2*highest_eig_value/np.pi)
                helper_eig = np.transpose(helper_eig)[0][0]

                centers_in_tot = np.zeros((2,3))
                centers_in_tot[0,:] = center_loop + helper_eig
                centers_in_tot[1,:] = center_loop - helper_eig

                centers_new, labels_new = kmeans(points_loop,2,1,max_singlerun_iterations,centers_in_tot)

                #split up the new centers into two parts:
                first_center = centers_new[0,:]
                second_center = centers_new[1,:]
                vector_v = first_center - second_center

                helper_skalar_product = np.dot(points_loop,vector_v)
                magnitude_v = np.sum(vector_v**2)
                projection = helper_skalar_product/magnitude_v

                ################################################################################################################
                # This part here would be necessary for an actual projection which would be achieved if we multiply projection
                # with the vector v in the end:

                # For a vector multiplication I first need to reshape the projection:
                #projection_x = projection.reshape((len(projection),1))
                #projection_x = projection_x*vector_v
                ################################################################################################################
                #Choose which center, new or old:
                estimation, critical, _ = anderson(projection)
                if estimation <= critical[-1] *tolerance:
                    is_gaussian = True
                    chosen_center_cluster = center_loop
                else:
                    chosen_center_cluster = centers_new

                # Now I want to check if those new clusters have points or not:


                if i == 0:
                    center_tot = chosen_center_cluster
                else:
                    center_tot = np.vstack((center_tot,chosen_center_cluster))

            elif points_loop.shape[0] == 0: # if cluster has zero points, center should be eliminated
                pass

            else: # if cluster has only one point, center should be kept!
                if i == 0:
                    center_tot = center_loop
                else:
                    center_tot = np.vstack((center_tot,center_loop))


        k = len(center_tot[:,0]) # New number of clusters
        chosen_centers = center_tot
        new_shape = len(chosen_centers[:, 0])


    # Though its not necessary in the end I start another k mean algorithm in order to get the labels:

    centers, labels = kmeans(points, k, 1, max_singlerun_iterations, chosen_centers)

    #centers = chosen_centers

    return centers, labels


def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """
    ######################################################
    number_of_points = points.shape[0]

    current_label = 0
    labels_total = np.zeros(number_of_points).astype(int)


    possible_points = np.arange(number_of_points)

    neighbour_points_random = np.array([])
    used_points = np.array([])

    #old_point = np.array([])
    #condition = False
    #while condition == False:

    # At the beginning I want to filter out all noise Points:
    for i in range(len(possible_points)):

        # Now I want to compute all distances from this random point:
        diff_point_filter = points - points[i, :]
        diff_point_squared_filter = diff_point_filter ** 2
        euclidian_distance_filter = np.sqrt(np.sum(diff_point_squared_filter, 1))

        # Now set zero distance to the max distance in order to avoid that the point i am looking at is considered a
        # neighbouring point:
        euclidian_distance_filter = np.where(euclidian_distance_filter == 0, np.max(euclidian_distance_filter), \
                                             euclidian_distance_filter)

        # Define all neighbouring points:
        neighbour_points_filter = np.where(euclidian_distance_filter <= eps)
        neighbour_points_filter = neighbour_points_filter[0]

        if len(neighbour_points_filter) <= min_samples:
            labels_total[i] = -1
            used_points = np.append(used_points, i)

    # This reduces my number of possible points
    possible_points = np.delete(possible_points,used_points)



    # The loop should continue as long as there are some points to look at!
    while len(possible_points) > 0:
        # Solange man Nachbarkpunkte findet, welche noch nicht benutz wurden sollen diese dem cluster hinzugefügt werden
        while len(neighbour_points_random) == 0 and len(possible_points) != 0:

            check_total = np.isin(possible_points,used_points,invert=True)
            possible_points = possible_points[check_total]

            # Choose an arbitrary possible point:
            if len(possible_points) > 0:
                random_select = np.random.choice(possible_points)

            # Now I want to compute all distances from this random point:
            diff_point_random = points - points[random_select,:]
            diff_point_squared_random = diff_point_random**2
            euclidian_distance_random = np.sqrt(np.sum(diff_point_squared_random,1))

            # Now set zero distance to the max distance in order to avoid that the point i am looking at is considered a
            # neighbouring point:
            euclidian_distance_random = np.where(euclidian_distance_random == 0, np.max(euclidian_distance_random), \
                                                 euclidian_distance_random)

            # Define all neighbouring points:
            neighbour_points_random = np.where(euclidian_distance_random <= eps)
            neighbour_points_random = neighbour_points_random[0]

        # Check whether random point is already part of the used points:

        next_point = random_select # for the first iteration of a cluster
        used_points = np.append(used_points,next_point)
        neighbour_points = neighbour_points_random # Needed in order to start the next loop

        while len(neighbour_points) > 0:

            # Now I want to compute all distances from the chosen point:
            diff_point = points - points[next_point,:]
            diff_point_squared = diff_point**2
            euclidian_distance = np.sqrt(np.sum(diff_point_squared,1))

            # Now set zero distance to the max distance in order to avoid that the point i am looking at is considered a
            # neighbouring point:
            euclidian_distance = np.where(euclidian_distance == 0, np.max(euclidian_distance), euclidian_distance)

            # Define all neighbouring points:
            neighbour_points = np.where(euclidian_distance <= eps)
            neighbour_points = neighbour_points[0]

            neighbour_points_original = neighbour_points


            # Now decide whether the selected point is a core point:
            if len(neighbour_points) > min_samples:
                labels_total[next_point] = current_label # Core Point
            else:
                labels_total[next_point] = -1 # Noise Point

            # Check if neighbour points have already been used and pass only the relevant points:
            helper_check = np.isin(neighbour_points, used_points, invert=True)
            neighbour_points = neighbour_points[helper_check].astype(int)

            '''
            ############################### DELETE !!!! ################################################################
            # Now give all neighbour points that fulfill the criteria also the current label:
            if len(neighbour_points)>0:
                neighbour_points_loop = neighbour_points
                for i in range(len(neighbour_points_loop)):

                    diff_point_loop = points - points[neighbour_points_loop[i], :]
                    diff_point_squared_loop = diff_point_loop ** 2
                    euclidian_distance_loop = np.sqrt(np.sum(diff_point_squared_loop, 1))

                    euclidian_distance_loop = np.where(euclidian_distance_loop == 0, np.max(euclidian_distance_loop),\
                                                       euclidian_distance_loop)

                    # Define all neighbouring points:
                    neighbour_points_helper_check = np.where(euclidian_distance_loop <= eps)
                    neighbour_points_helper_check = neighbour_points_helper_check[0]

                    helper_check_loop = np.isin(neighbour_points_loop, used_points, invert=True)
                    neighbour_points_loop = neighbour_points_loop[helper_check_loop].astype(int)

                    if len(neighbour_points_helper_check) > min_samples:
                        labels_total[neighbour_points_loop[i]] = current_label
                    else:
                        labels_total[next_point] = -1  # Noise Point

            ############################################################################################################
            '''



            if len(neighbour_points) > 0:  # adapted

                next_point = neighbour_points[0]
                helper_check_next_point = np.isin(next_point, used_points)

                counter = 0
                while helper_check_next_point == True and counter < (len(neighbour_points) - 1):
                    counter += 1
                    next_point = neighbour_points[counter]
                    helper_check_next_point = np.isin(next_point, used_points)
                    print(counter)

                if helper_check_next_point == False:
                    used_points = np.append(used_points, next_point)
            else:
                current_label += 1
                possible_points = np.delete(possible_points,used_points)
                neighbour_points_random = neighbour_points





        ################################################################################################################


        '''

        # Now check if all neighbour points are core points or can be considered as noise only if there are still points
        if len(neighbour_points_original) > 0: #adapted
            for i in range(len(neighbour_points)):
                # Again calculation of the euclidian distance:
                diff_point_loop = points - points[neighbour_points[i], :]
                diff_point_squared_loop = diff_point_loop ** 2
                euclidian_distance_loop = np.sqrt(np.sum(diff_point_squared_loop, 1))
                euclidian_distance_loop = np.where(euclidian_distance_loop == 0, np.max(euclidian_distance_loop), euclidian_distance_loop)

                neighbour_points_loop = np.where(euclidian_distance_loop <= eps)
                neighbour_points_loop = neighbour_points_loop[0]

                if len(neighbour_points_loop) > min_samples:
                    labels_total[neighbour_points[i]] = current_label  # Core Point
                    used_points = np.append(used_points, neighbour_points[i])

                else:
                    labels_total[neighbour_points[i]] = -1  # Noise Point
                    used_points = np.append(used_points, neighbour_points[i])

            # Now choose a next point and start with the first neighbour
            diff_point_next = points - points[neighbour_points_original[0], :]
            diff_point_squared_next = diff_point_next ** 2
            euclidian_distance_next = np.sqrt(np.sum(diff_point_squared_next, 1))
            euclidian_distance_next = np.where(euclidian_distance_next == 0, np.max(euclidian_distance_next), euclidian_distance_next)

            neighbour_points_next = np.where(euclidian_distance_next <= eps)
            neighbour_points_next = neighbour_points_next[0]

            next_point = neighbour_points_next[0]

            helper_check_next_point = np.isin(next_point, used_points)

            counter = 0

            while helper_check_next_point == True and counter < (len(neighbour_points_next)-1):
                counter += 1
                next_point = neighbour_points_next[counter]
                helper_check_next_point = np.isin(next_point, used_points)

            if helper_check_next_point == False:
                used_points = np.append(used_points, next_point)
            else:
                pass
                #current_label += 1
                #print(next_point)
                #print(current_label)
                #print('Happened in first AFTER WHILE')


        else:
            current_label += 1
            possible_points = np.delete(possible_points,used_points)
        
        '''

    labels = labels_total
    #labels = np.zeros(shape=(len(points),), dtype=int)

    return labels
