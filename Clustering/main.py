
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


from fit_plane import fit_plane
from clustering import *
from helper_functions import *

from sklearn.cluster import DBSCAN # To check

if __name__ == '__main__':

    # Selects which single-plane file to use
    pointcloud_idx = 0 #Original: 0

    # Pick which clustering algorithm to apply:
    use_kmeans = False
    use_iterative_kmeans = False
    use_gmeans = True
    use_dbscan = True

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015  # Might need to be adapted, depending on how you implement fit_plane

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.01
    uniform_every_k_points = 10

    # Clustering Parameters
    kmeans_n_clusters = 6
    kmeans_iterations = 25 # Original value: 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10 # 10
    gmeans_tolerance = 10
    dbscan_eps = 0.05 # 0.05
    dbscan_min_points = 15 # 15
    debug_output = True

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("pointclouds/image00")) + str(pointcloud_idx) + ".pcd",
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

    ####################################### ADDED FOR DOCUMENTATION ####################################################
    '''
    dataset_sizes = np.array([1614, 1668, 1565, 1742, 1585, 1666, 1747, 1716, 1670, 2404])
    dataset_sizes = np.sort(dataset_sizes)
    dataset_sizes_adapted = dataset_sizes

    for i in range(len(dataset_sizes)-1):
        if dataset_sizes[i+1]-dataset_sizes[i] < 10:
            dataset_sizes_adapted = np.delete(dataset_sizes_adapted,i)
        else:
            pass

    dataset_sizes = dataset_sizes_adapted
    '''
    ####################################################################################################################

    # k-Means
    if use_kmeans:
        # Apply k-means algorithm
        centers, labels = kmeans(points,
                                 n_clusters=kmeans_n_clusters,
                                 n_iterations=kmeans_iterations,
                                 max_singlerun_iterations=max_singlerun_iterations)

        plot_clustering_results(scene_pcd,
                                labels,
                                "K-means")


    # Iterative k-Means
    if use_iterative_kmeans:
        centers, labels = iterative_kmeans(points,
                                           max_n_clusters=iterative_kmeans_max_clusters,
                                           n_iterations=kmeans_iterations,
                                           max_singlerun_iterations=max_singlerun_iterations)
        plot_clustering_results(scene_pcd,
                                labels,
                                "Iterative k-means")

    # G-Means
    if use_gmeans:
        centers, labels = gmeans(points,
                                 tolerance=gmeans_tolerance,
                                 max_singlerun_iterations=max_singlerun_iterations)
        plot_clustering_results(scene_pcd,
                                labels,
                                "G-means")
    # DBSCAN
    if use_dbscan:
        '''
        # I implemented this just in case if i run into problems, but I was able to fix the code so my dbscan hopefully
        #works fine now:
        
        cluster_dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points).fit(points)
        labels = cluster_dbscan.labels_
        '''

        labels = dbscan(points,
                        eps=dbscan_eps,
                        min_samples=dbscan_min_points)

        plot_clustering_results(scene_pcd,
                                labels,
                                "DBSCAN")




