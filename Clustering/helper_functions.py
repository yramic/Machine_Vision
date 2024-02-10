#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results and calculate the silhouette score
Author: Ramic Yannick
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import distance


def silhouette_score(points: np.ndarray,
                     centers: np.ndarray,
                     labels: np.ndarray) -> float:
    """ Calculate the silhouette score for clusters.

    Calculate the silhouette score for clusters. It specifies how similar a sample is to its own cluster compared to
    other clusters. It was defined in Rousseeuw, Peter J. "Silhouettes: a graphical aid to the interpretation and
    validation of cluster analysis." J. Comput. Appl. Math. 20 (1987): 53-65.

    :param points: The (down-sampled) points of the pointcloud
    :type points: np.ndarray with shape = (n_points, 3)

    :param centers: The centers of the clusters. Each row is the center of a cluster.
    :type centers: np.ndarray with shape = (n_clusters, 3)

    :param labels: Array with a different label for each cluster for each point
            The label i corresponds with the center in centers[i]
    :type labels: np.ndarray with shape = (n_points,)

    :return: The calculated silhouette score is the mean silhouette coefficient of all samples in the range [-1, 1]
    :rtype: float
    """
    ######################################################
    # First some description of the task:
    # The center points are needed in order to locate the nearest cluster
    # 1) Each point has a value a (Average of all distances to all points in the same cluster)
    # 2) Each point has a value b (Average of all distances to all points corresponding to the nearest cluster)

    # First I want to assign for each cluster the nearest neighbouring cluster:

    n_points = points.shape[0]
    n_clusters = centers.shape[0]
    nearest_neighbours = np.zeros((1,n_clusters))[0] # For each cluster the closest cluster should be stored here

    ####################################################################################################################
    #for i in range(n_clusters):
        #diff_helper = (centers - centers[i])**2
        #diff_sum = np.sum(diff_helper,1)
        #diff_sum = np.where(diff_sum==0,100,diff_sum) #The value zero needs to be set high in order to find the closest result
        #nearest_cluster = np.nonzero(diff_sum == np.min(diff_sum)) # This gives the back the closest neighbour
        #nearest_neighbours[i] = nearest_cluster[0]
    #nearest_neighbours = nearest_neighbours.astype(int)
    ####################################################################################################################

    # The nearest neighbour is only relevant to calculate b, for a it is not needed

    # Next step is to calculate a and b therefore I need a loop over every point:
    a = np.zeros(n_points)
    b = np.zeros(n_points)
    s = np.zeros(n_points)

    # For loop should only run if the cluster contains more than one point!
    for j in range(n_points):
        # Approach for a:
        # Which cluster is each point assigned to:
        helper_cluster = labels[j]
        # Now find every point with that is assigned to the same cluster:
        idx_cluster_a = np.nonzero(labels == helper_cluster) # These are the rows for each point!
        points_total_cluster_a = len(idx_cluster_a[0])

        if points_total_cluster_a > 1 and n_clusters > 1:
            # When we calculate the euclidian difference it is important to note that the point j will become zero and has
            # no influence on the sum in the end
            helper_a_0 = points[idx_cluster_a,:] - points[j,:]
            helper_a = helper_a_0 ** 2
            d_euclidian_a_squared = np.sum(helper_a,1) # Gives the euclidian distance for each point
            d_euclidian_a = np.sqrt(d_euclidian_a_squared)
            a[j] = np.sum(d_euclidian_a) / (points_total_cluster_a-1)

            # Same approach now for b:
            # First I want to check which is the next closest center to the point j:
            closest_cluster_helper = centers-points[j,:]
            closest_cluster = closest_cluster_helper ** 2
            closest_cluster = np.sum(closest_cluster,1)
            closest_cluster = np.sqrt(closest_cluster)

            find_closest_cluster = np.sort(closest_cluster)
            nearest_cluster_b = find_closest_cluster[1]
            idx_nearest_cluster_b = np.nonzero(closest_cluster == nearest_cluster_b)
            idx_nearest_cluster_b = idx_nearest_cluster_b[0][0]

            helper_cluster_b = idx_nearest_cluster_b


            #helper_cluster_b = nearest_neighbours[helper_cluster] # Gives back the nearest cluster
            idx_cluster_b = np.nonzero(labels == helper_cluster_b)
            points_total_cluster_b = len(idx_cluster_b[0])
            helper_b_0 = points[idx_cluster_b,:] - points[j,:]
            helper_b = helper_b_0 ** 2
            d_euclidian_b_squared = np.sum(helper_b,1)
            d_euclidian_b = np.sqrt(d_euclidian_b_squared)
            b[j] = np.sum(d_euclidian_b) / points_total_cluster_b

            # Now we can calculate the score s for each value:
            if a[j] >= b[j]:
                max = a[j]
                s[j] = ((b[j] - a[j]) / max)
            else:
                max = b[j]
                s[j] = ((b[j] - a[j]) / max)

        elif points_total_cluster_a == 1:
            a[j] = 0
            b[j] = 0
            s[j] = 0
        elif n_clusters == 1:
            a[j] = 0
            b[j] = 0
            s[j] = -1


    # The score should now be the mean over all values s for each point:
    score = np.mean(s)

    return score

def plot_dominant_plane(pcd: o3d.geometry.PointCloud,
                        inliers: np.ndarray,
                        plane_eq: np.ndarray) -> None:
    """ Plot the inlier points in red and the rest of the pointcloud as is. A coordinate frame is drawn on the plane

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.array

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.array [a,b,c,d]

    :return: None
    """

    # Filter the inlier points and color them red
    inlier_indices = np.nonzero(inliers)[0]
    inlier_cloud = pcd.select_by_index(inlier_indices)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    # Create a rotation matrix according to the plane equation.
    # Detailed explanation of the approach can be found here: https://math.stackexchange.com/q/1957132
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    u2 = np.cross(normal_vector, u1)
    rot_mat = np.c_[u1, u2, normal_vector]

    # Create a coordinate frame and transform it to a point on the plane and with its z-axis in the same direction as
    # the normal vector of the plane
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.rotate(rot_mat, center=(0, 0, 0))
    if any(inlier_indices):
        coordinate_frame.translate(np.asarray(inlier_cloud.points)[-1])
        coordinate_frame.scale(0.3, np.asarray(inlier_cloud.points)[-1])

    geometries = [inlier_cloud, outlier_cloud, coordinate_frame]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.3, 0.32, -0.9])
    vc.set_lookat([-0.13, -0.15, 0.92])
    vc.set_up([0.22, -0.89, -0.39])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def plot_clustering_results(pcd: o3d.geometry.PointCloud,
                            labels: np.ndarray,
                            method_name: str):
    labels = labels - labels.min()
    print(method_name + f": Point cloud has {int(labels.max()) + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

# From https://github.com/isl-org/Open3D/issues/2
def text_3d(text, pos, direction=None, degree=0.0, font='RobotoMono-Medium.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd