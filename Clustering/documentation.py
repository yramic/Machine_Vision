#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

###############################

from fit_plane import fit_plane
from clustering import *
from helper_functions import *

from sklearn.cluster import DBSCAN # To check

if __name__ == '__main__':

    # Selects which single-plane file to use
    pointcloud_idx = 9 #Original: 0

    # Pick which clustering algorithm to apply:
    use_kmeans = True
    use_iterative_kmeans = True
    use_gmeans = True
    use_dbscan = True

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015  # Might need to be adapted, depending on how you implement fit_plane

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.01 # 0.01
    uniform_every_k_points = 10 # 10

    # Clustering Parameters
    kmeans_n_clusters = 6
    kmeans_iterations = 25 # Original value: 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10 # 10
    gmeans_tolerance = 10
    dbscan_eps = 0.05
    dbscan_min_points = 15
    debug_output = False


    store_time_kmeans = np.zeros(10)
    store_time_kmeans_it = np.zeros(10)
    store_time_gmeans = np.zeros(10)
    store_time_dbscan = np.zeros(10)


    # Read Pointcloud
    for i in range(10):
        current_path = Path(__file__).parent
        pcd = o3d.io.read_point_cloud(str(current_path.joinpath("pointclouds/image00")) + str(i) + ".pcd",
                                      remove_nan_points=True, remove_infinite_points=True)
        if not pcd.has_points():
            raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

        # Down-sample the loaded point cloud to reduce computation time
        if use_voxel_downsampling:
            pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        else:
            pcd_sampled = pcd.uniform_down_sample(uniform_every_k_points)

        # Apply your own plane-fitting algorithm
        # plane_model, best_inliers = fit_plane(pcd=pcd_sampled,
        #                                       confidence=confidence,
        #                                       inlier_threshold=inlier_threshold)
        # inlier_indices = np.nonzero(best_inliers)[0]

        # Alternatively use the built-in function of Open3D
        plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
                                                                ransac_n=3,
                                                                num_iterations=500)

        # Convert the inlier indices to a Boolean mask for the pointcloud
        best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
        best_inliers[inlier_indices] = True

        # Store points without plane in scene_pcd
        scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

        # Plot detected plane and remaining pointcloud
        if debug_output:
            plot_dominant_plane(pcd_sampled, best_inliers, plane_model)
            o3d.visualization.draw_geometries([scene_pcd])

        # Convert to NumPy array
        points = np.asarray(scene_pcd.points, dtype=np.float32)

        count_time_k_means = time.time()
        # k-Means
        if use_kmeans:
            # Apply k-means algorithm
            centers, labels = kmeans(points,
                                     n_clusters=kmeans_n_clusters,
                                     n_iterations=kmeans_iterations,
                                     max_singlerun_iterations=max_singlerun_iterations)

            #plot_clustering_results(scene_pcd,
                                    #labels,
                                    #"K-means")

        store_time_kmeans[i] = (time.time() - count_time_k_means)

            # Set up for time analysis:
            #k_means_test = sklearn.cluster.KMeans(n_clusters=kmeans_n_clusters)
            #k_means_data = benchmark_algorithm(dataset_sizes, k_means_test.fit, (), {})

        # Iterative k-Means
        count_time_k_means_it = time.time()
        if use_iterative_kmeans:
            centers, labels = iterative_kmeans(points,
                                               max_n_clusters=iterative_kmeans_max_clusters,
                                               n_iterations=kmeans_iterations,
                                               max_singlerun_iterations=max_singlerun_iterations)
            #plot_clustering_results(scene_pcd,
                                    #labels,
                                    #"Iterative k-means")
        store_time_kmeans_it[i] = (time.time() - count_time_k_means_it)



        # G-Means
        count_time_g_means = time.time()
        if use_gmeans:
            centers, labels = gmeans(points,
                                     tolerance=gmeans_tolerance,
                                     max_singlerun_iterations=max_singlerun_iterations)
            #plot_clustering_results(scene_pcd,
                                    #labels,
                                    #"G-means")
        store_time_gmeans[i] = (time.time() - count_time_g_means)

        # DBSCAN
        count_time_dbscan = time.time()
        if use_dbscan:
            # I implemented this just in case if i run into problems, but I was able to fix the code so my dbscan hopefully
            # works fine now:

            #cluster_dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points).fit(points)
            #labels = cluster_dbscan.labels_

            labels = dbscan(points,
                            eps=dbscan_eps,
                            min_samples=dbscan_min_points)
            #plot_clustering_results(scene_pcd,
                                    #labels,
                                    #"DBSCAN")
        store_time_dbscan[i] = (time.time() - count_time_dbscan)

print(store_time_dbscan)

x_achsis = np.arange(0, 10, 1) #pointclouds


plt.plot(x_achsis, store_time_kmeans, color='r', label='k-means')
plt.plot(x_achsis, store_time_kmeans_it, color='g', label='iterative k-means')
plt.plot(x_achsis, store_time_gmeans, color='b', label='g-means')
plt.plot(x_achsis, store_time_dbscan, color='y', label='dbscan')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Different Pointclouds")
plt.ylabel("CPU [s]")
plt.title("Comparison")

# Adding legend, which helps us recognize the curve according to it's color
#plt.legend()
plt.savefig('All_Runtimes.png')

# To load the display window
plt.show()
