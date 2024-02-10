"""
Yannick Ramic 2022
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from helper_functions import *
from sklearn.cluster import DBSCAN
from camera_params import *
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
from glob import glob

if __name__ == '__main__':

    '''
    Note: The first part was copied from Exercise 4 to plot the point-cloud.
    '''

    # Selects which single-plane file to use
    pointcloud_idx = 0

    # Select object for the matching process to use for the relevant test dataset:
    point_cloud_idx_test = 0

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015  # Might need to be adapted, depending on how you implement fit_plane

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.002 #original: 0.01
    uniform_every_k_points = 10

    # Clustering Parameter:
    use_dbscan = True
    dbscan_eps = 0.05 # 0.05
    dbscan_min_points = 300 # NOTE! This number needs to be reduced if voxel size becomes higher!

    debug_2d_picture = False # Only possible to change while use_dbscan is active

    # delete cluster with less than 20 points!
    constraint_cluster = 50

    # Debug Parameter:
    debug_output = False # Set False to neglect the plot for the point-cloud

    # Show Image:
    show_images = False
    show_images_cluster = True
    show_images_object = False

    # SIFT Parameters:
    use_sift = True
    debug_sift = False # Pay attention because it is inside the loop, otherwise change the loop or set breakpoints!
    store_analysis_df = False # This safes the dataframe for further evaluation as a csv file
    store_df = False
    store_img = False

    # Read Point-cloud
    current_path = Path(__file__).parent

    # This gives back the point cloud for the scene image:
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image00")) + str(pointcloud_idx) + ".pcd",
                                  remove_nan_points=True, remove_infinite_points=True)

    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load scene point cloud in " + str(current_path))


    # Down-sample the loaded point cloud to reduce computation time
    if use_voxel_downsampling:
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_sampled = pcd.uniform_down_sample(uniform_every_k_points)

    # Only pcd_scene requires a down sampling, this is not relevant for the object!
    # Also the RANSAC algorithm is not necessary for the object!

    ####################################################################################################################
    ########################################## 1) RANSAC ###############################################################
    ####################################################################################################################
    # Due to the fact, that I didn't get all points for the RANSAC Algorithm, I decided to implement the Open3D's
    # segment_plane function:
    plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
                                                            ransac_n=3,
                                                            num_iterations=500)

    # Convert the inlier indices to a Boolean mask for the point-cloud
    best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
    best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)
    scene_adapted_pcd = pcd_sampled.select_by_index(inlier_indices,invert=True)
    # Due to the fact that I do not want to overwrite later on scene_pcd, I created another point cloud that is equal to
    # scene_pcd, called scene_adapted_pcd


    # Plot detected plane and remaining point-cloud
    if debug_output:
        plot_dominant_plane(pcd_sampled, best_inliers, plane_model)
        o3d.visualization.draw_geometries([scene_pcd])

    # Convert to NumPy array
    points = np.asarray(scene_pcd.points, dtype=np.float32)
    colors = np.asarray(scene_pcd.colors) # Note: BGR needs to converted to RGB!
    colors = colors[:,::-1] # Here are the colors for each point stored as RGB!

    ####################################################################################################################
    ########################################## 2) 2D Image Space #######################################################
    ####################################################################################################################

    points_2D_original = projection_3d_to_2d(points_3d = pcd)
    points_2D_scene = projection_3d_to_2d(points_3d = scene_pcd)

    ####################################################################################################################
    ########################################## 3) DBSCAN ###############################################################
    ####################################################################################################################
    # Next I will use the dbscan algorithm to cluster the point cloud:
    # Note: Obviously this only has to be done for the scenery and not for the object!
    if use_dbscan:

        cluster_dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points).fit(points)
        labels = cluster_dbscan.labels_

        keep_points = np.where(labels > np.min(labels))
        keep_points = keep_points[0]
        labels_total_adapted = labels[keep_points]

        scene_pcd = scene_pcd.select_by_index(keep_points)

        # Also the colors need to be changed:
        colors = colors[keep_points,:]
        points = points[keep_points,:]
        points_2D_scene = points_2D_scene[keep_points,:]

        # This was my first approach
        '''
        # Now I want to delete all clusters with little points!
        # The following steps result in a point cloud with reduced clusters! (Relevant Parameter: constraint_cluster)
        unique_labels = np.unique(labels)
        delete_labels = np.array([])
        delete_points = np.array([])
        keep_labels = np.array([])
        keep_points = np.array([])

        for i in unique_labels:
            if np.count_nonzero(labels == i) <= constraint_cluster:
                delete_labels = np.append(delete_labels, i).astype(int)
            else:
                keep_labels = np.append(keep_labels, i).astype(int)

        if len(delete_labels > 0):
            for i in delete_labels:
                delete_points_i = np.nonzero(labels == i)
                delete_points = np.append(delete_points, delete_points_i)

            for j in keep_labels:
                keep_points_j = np.nonzero(labels == j)
                keep_points = np.append(keep_points, keep_points_j)

            delete_points = delete_points.astype(int)
            keep_points = keep_points.astype(int)

        labels_total_adapted = labels[keep_points]

        scene_adapted_pcd = scene_adapted_pcd.select_by_index(keep_points)

        
        points_total_scene_pcd = np.linspace(0,len(points)-1,len(points),dtype=int)
        helper_logic_points = np.isin(points_total_scene_pcd, delete_points)
        helper_label_points = np.isin(labels, keep_labels)
        points_total_adapted = np.nonzero(helper_logic_points == False)
        points_idx_adapted = points_total_adapted[0]
        labels_total_adapted = np.nonzero(helper_label_points == True)
        labels_idx_adapted = labels_total_adapted[0]

        points_adapted = points[points_idx_adapted,:]
        labels_adapted = labels[labels_idx_adapted]


        point_cloud_adapted = scene_adapted_pcd
        point_cloud_adapted.points = o3d.utility.Vector3dVector(points_adapted)

        # Now I need to overwrite the DBSCAN again for a correct implementation, Otherwise unfortunately the code does
        # not work!
        cluster_dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points).fit(points_adapted)
        labels = cluster_dbscan.labels_
        '''

        if show_images_cluster:
            plot_clustering_results(scene_pcd,
                                    labels_total_adapted,
                                    "DBSCAN")

    # Convert to NumPy array
    points_2D_new = projection_3d_to_2d(points_3d=scene_pcd)
    # Note: the following color sample is just from the labels!
    colors_new = np.asarray(scene_pcd.colors)  # Note: BGR needs to converted to RGB!
    colors_new = colors_new[:, ::-1]  # Here are the colors for each point stored as RGB!

    '''
    # Another approach to show the 2D image:
    # Image with holes compared to figure 3 (left side) of the exercise sheet:
    plt.scatter(points_2D_new[:, 0], points_2D_new[:, 1], c=labels, cmap='jet')
    plt.show()
    '''

    # Note: that the coordinates are as followed for the image: y,x and not x,y!!!
    x_min = 0
    x_max = np.max(points_2D_scene[:,0].astype(int))
    width_img = x_max - x_min

    y_min = 0
    y_max = np.max(points_2D_scene[:,1].astype(int))
    height_img = y_max - y_min

    '''
    # This commented code lines are just a test for a black and white image!
    image_2D = np.zeros(shape=(height_img+1, width_img+1))

    u_coordinates = points_2D_new[:,0].astype(int)
    v_coordinates = points_2D_new[:,1].astype(int)

    image_2D[v_coordinates,u_coordinates] = 1

    
    ## Another approach to fill the values:
    #for point in points_2D_new:
    #    y, x = point.astype(int)
    #    if x >= 0 and x < width_img and y >= 0 and y < height_img:
    #        image_2D[y, x] = 1
    
    # Next I want to apply a morphological operator:
    # Use OpenCV to fill holes in the image
    kernel = np.ones((3, 3), np.uint8)
    dilated_coordinates = cv2.dilate(image_2D, kernel, iterations=10)
    # Note that dilated coordinates results in a black and white image because the values are eater 1 or zero!
    '''

    # Now I want to add the underlying color to the image:
    # Therefore I first need the shape:
    colors_range = colors*255
    colors_range = colors_range.astype(np.dtype(np.uint8))
    image_2d_rgb = np.zeros(shape=(height_img+1, width_img+1,3))

    for i in points_2D_scene:
        idx_point = np.nonzero(i == points_2D_scene)
        actual_point = idx_point[0][0]
        x, y = i.astype(int)
        if x >= 0 and x < width_img and y >= 0 and y < height_img:
            image_2d_rgb[y, x, :] = colors_range[actual_point,:]

    image_2d_rgb = image_2d_rgb.astype(np.dtype(np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    dilated_coordinates_rgb = cv2.dilate(image_2d_rgb, kernel, iterations=5)


    if debug_2d_picture:
        colors_range_debug = colors_new*255
        colors_range_debug = colors_range_debug.astype(int)
        image_2d_rgb_debug = np.zeros(shape=(height_img + 1, width_img + 1, 3))

        for i_debug in points_2D_new:
            idx_point_debug = np.nonzero(i_debug == points_2D_new)
            actual_point_debug = idx_point_debug[0][0]
            x_debug, y_debug = i_debug.astype(int)
            if x_debug >= 0 and x_debug < width_img and y_debug >= 0 and y_debug < height_img:
                image_2d_rgb_debug[y_debug, x_debug, :] = colors_range_debug[actual_point_debug, :]

        image_2d_rgb_debug = image_2d_rgb.astype(np.uint8)
        kernel_debug = np.ones((3, 3), np.uint8)
        dilated_coordinates_rgb_debug = cv2.dilate(image_2d_rgb_debug, kernel_debug, iterations=5)

        dilated_coordinates_rgb_debug = dilated_coordinates_rgb_debug.astype(int)

        plt.imshow(dilated_coordinates_rgb_debug)
        plt.show()


    # Test 2D projection, show_images is a parameter above and this has to be true!
    if show_images:
        ## This produces a black and white image to test whether the 2D projection has worked
        #plt.imshow(dilated_coordinates, cmap='gray')
        #plt.show()
        # This produces a coloured image with a colour jet!
        plt.imshow(image_2d_rgb)
        plt.show()
        #use_matplotlib = True
        #show_image(img= dilated_coordinates_rgb, title="debug", use_matplotlib=use_matplotlib)

    ####################################################################################################################
    ############################################## SIFT ################################################################
    ####################################################################################################################

    # Before I start the loop for the SIFT voting process, I want to have a grayscale image of the scenery:
    scene_gray = cv2.cvtColor(image_2d_rgb, cv2.COLOR_RGB2GRAY)

    if use_sift:

        current_path = Path(__file__).parent
        folder_path = current_path

        folder_path = folder_path.joinpath("training")
        files = glob(os.path.join(folder_path, '*'))

        file_data = {'file_name': [], 'cluster_matches_max': [], 'number_total': [], 'accuracy': []}

        counter_iterations = 0
        for file in files:
            print('Training Dataset: ', counter_iterations)
            file_name = os.path.basename(file)
            file_data['file_name'].append(file_name)

            object_pcd = o3d.io.read_point_cloud(str(folder_path.joinpath(file_name)), remove_nan_points = True,
                                                 remove_infinite_points = True)
            if not object_pcd.has_points():
                raise FileNotFoundError("Couldn't load object point cloud in " + str(current_path))

            # Same processes now for the object:
            points_object = np.asarray(object_pcd.points, dtype=np.float32)
            colors_object = np.asarray(object_pcd.colors)
            colors_object = colors_object[:, ::-1]

            points_2D_object = projection_3d_to_2d(points_3d=object_pcd)

            # Same thing for the object:
            x_min_object = 0
            x_max_object = np.max(points_2D_object[:, 0].astype(int))
            width_img_object = x_max_object - x_min_object

            y_min_object = 0
            y_max_object = np.max(points_2D_object[:, 1].astype(int))
            height_img_object = y_max_object - y_min_object

            # Again same thing for the object:
            colors_range_object = colors_object * 255
            colors_range_object = colors_range_object.astype(np.dtype(np.uint8))
            image_2d_rgb_object = np.zeros(shape=(height_img_object + 1, width_img_object + 1, 3))

            for j in points_2D_object:
                idx_point_object = np.nonzero(j == points_2D_object)
                actual_point_object = idx_point_object[0][0]
                x_object, y_object = j.astype(int)
                if x_object >= 0 and x_object < width_img_object and y_object >= 0 and y_object < height_img_object:
                    image_2d_rgb_object[y_object, x_object, :] = colors_range_object[actual_point_object, :]

            image_2d_rgb_object = image_2d_rgb_object.astype(np.dtype(np.uint8))
            dilated_coordinates_rgb_object = cv2.dilate(image_2d_rgb_object, kernel, iterations=5) # Only for plotting!!!

            if show_images_object:
                plt.imshow(image_2d_rgb_object)
                plt.show()

            # First I want to convert both images to grayscale!
            object_gray = cv2.cvtColor(image_2d_rgb_object, cv2.COLOR_RGB2GRAY)

            # Get SIFT keypoints_1 and descriptors
            sift = cv2.SIFT_create()
            target_keypoints, target_descriptors = sift.detectAndCompute(scene_gray, None)
            source_keypoints, source_descriptors = sift.detectAndCompute(object_gray, None)

            # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # For each keypoint in object_img get the 3 best matches
            matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)

            matches = filter_matches(matches)

            if debug_sift:
                draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                matches_img = cv2.drawMatches(image_2d_rgb_object,
                                              source_keypoints,
                                              image_2d_rgb,
                                              target_keypoints,
                                              matches,
                                              None,
                                              **draw_params)
                show_image(matches_img, "Matches", save_image=False, use_matplotlib=True)

            adjacent_pixels_y = np.array([])
            adjacent_pixels_x = np.array([])

            for match in matches:
                train_idx = match.trainIdx
                query_idx = match.queryIdx

                # Get the 2D coordinates of the matched keypoints
                train_coord = source_keypoints[query_idx].pt
                query_coord = target_keypoints[train_idx].pt # target = scene_image

                # Get the adjacent pixel of the matched 2D coordinates
                # I added the one because afterwards I need integers and converting it to integers it will be rounded, as a
                # result to somehow compensate that I added the value one to each direction!
                train_adjacent_pixel = (train_coord[0] + 1, train_coord[1] + 1)
                query_adjacent_pixel = (query_coord[0] + 1, query_coord[1] + 1) # Relevant Data for the scene

                # Append the adjacent pixels to the list
                #adjacent_pixels = np.append(train_adjacent_pixel, query_adjacent_pixel)
                adjacent_pixels_x = np.append(adjacent_pixels_x, query_adjacent_pixel[0])
                adjacent_pixels_y = np.append(adjacent_pixels_y, query_adjacent_pixel[1])

            # Next I want to check to what cluster the adjacent pixel corresponds:
            adjacent_pixels_x = adjacent_pixels_x.astype(int)
            adjacent_pixels_y = adjacent_pixels_y.astype(int)

            adjacent_pixels = np.array([adjacent_pixels_x,adjacent_pixels_y])
            adjacent_pixels = np.transpose(adjacent_pixels)

            # I need to compare them with points_2D_scene and the underlying labels_total_adapted:
            test_points_scene = points_2D_scene.astype(int)

            # I want to store for all the matches the number of clusters it hit each time in the following way:
            store_results = np.zeros(shape=(2,len(np.unique(labels_total_adapted))))
            store_results = store_results.astype(int)
            store_results[0,:] = np.unique(labels_total_adapted).astype(int)

            for row_px in range(len(adjacent_pixels[:,0])):
                difference_px = test_points_scene - adjacent_pixels[row_px]
                difference_px = abs(difference_px)
                diff_sum_px = np.sum(difference_px, axis=1)
                idx_px = np.nonzero(diff_sum_px == np.min(diff_sum_px))
                # Now that I know the index the match is, I can evaluate which label it has:
                corresponding_label = labels_total_adapted[idx_px]
                store_results[1,corresponding_label] += 1

            # Now I want to evaluate the cluster that has the most matches:
            cluster_max = np.nonzero(store_results[1,:] == np.max(store_results[1,:]))
            cluster_max = cluster_max[0][0]

            # This gives back the number of how many matches are in cluster_max:
            number_matches = store_results[1,cluster_max]

            # Now I need to come up with some sort of NORMALIZATION and I choose to divide the number_matches through
            # the total number of matches.
            # Note: This is one of the easiest forms of normalization F1-score or other methods are probably better

            accuracy = number_matches/len(matches)
            accuracy = round(accuracy, 2) # round to the second decimal!

            file_data['cluster_matches_max'].append(cluster_max)
            file_data['number_total'].append(number_matches)
            file_data['accuracy'].append(accuracy)

            counter_iterations += 1

        df = pd.DataFrame(file_data)

        if store_df:
            # Specify the file name and path
            file_name_df = 'total_table.csv'
            # Write the DataFrame to a CSV file
            df.to_csv(file_name_df, index=False)

        unique_cluster_matches = np.unique(file_data['cluster_matches_max'])

        analysis_data = {'cluster': [], 'accuracy_max': [], 'file_name': []}

        for find_max in unique_cluster_matches:
            idx_cluster = np.nonzero(file_data['cluster_matches_max'] == unique_cluster_matches[find_max])
            idx_cluster = idx_cluster[0]
            store_accuracies = list(map(file_data['accuracy'].__getitem__, idx_cluster))
            accuracy_max = np.max(store_accuracies)
            idx_accuracy = np.nonzero(store_accuracies == accuracy_max)
            idx_accuracy = idx_accuracy[0][0]
            idx_cluster_max = idx_cluster[idx_accuracy]

            analysis_data['accuracy_max'].append(accuracy_max)
            analysis_data['cluster'].append(find_max)
            analysis_data['file_name'].append(file_data['file_name'][idx_cluster_max])

        df_analysis = pd.DataFrame(analysis_data)
        if store_analysis_df:
            # Specify the file name and path
            file_name_analysis = 'analysis_table.csv'
            # Write the DataFrame to a CSV file
            df_analysis.to_csv(file_name_analysis, index=False)

        # Convert the numpy array to a PIL Image
        image_result = Image.fromarray(np.uint8(image_2d_rgb))

        # Create an ImageDraw object
        draw = ImageDraw.Draw(image_result)

        # Define the font and font size
        font = ImageFont.truetype("arial.ttf", 16)

        # To define the position I first need to get a cluster analysis. I have the points and the cluster and I want to
        # have the text in the center of each cluster:

        dic_drawing = {'cluster': [], 'x': [], 'y': []}
        unique_labels = np.unique(labels_total_adapted)
        for idx_unique_label in unique_labels:
            idx_label = np.nonzero(labels_total_adapted == idx_unique_label)
            idx_label = idx_label[0]

            x_min = np.min(test_points_scene[idx_label,0])
            x_max = np.max(test_points_scene[idx_label,0])
            x_middle = (x_min+x_max)/2
            x_middle = x_middle.astype(int)

            y_min = np.min(test_points_scene[idx_label,1])
            y_max = np.max(test_points_scene[idx_label,1])
            y_middle = (y_min+y_max)/2
            y_middle = y_middle.astype(int)

            dic_drawing['cluster'].append(idx_unique_label)
            dic_drawing['x'].append(x_middle)
            dic_drawing['y'].append(y_middle)

        df_drawing = pd.DataFrame(dic_drawing)

        # Define the text and its position
        text_data = analysis_data['file_name']
        text_adapted = [text[:-7] for text in text_data]
        text_data = text_adapted

        relevant_clusters = analysis_data['cluster']

        x_px = []
        y_px = []
        for i_cluster in relevant_clusters:
            x_px.append(dic_drawing['x'][i_cluster])
            y_px.append(dic_drawing['y'][i_cluster])

        # Draw a red text (255, 0, 0) on the image
        for x, y, text in zip(x_px, y_px, text_data):
            draw.text((x, y), text, font=font, fill=(255, 0, 0))

        if store_img:
            image_result.save("image_2d_rgb_1.png")

        image_result.show()










