# Calibrated using the RGBDemo Calibration tool:
#   http://rgbdemo.org/
# 

import numpy as np
import open3d as o3d

# The maximum depth used, in meters.
max_depth = 10

# RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02 - 5
cy_rgb = 2.5373616633400465e+02 - 12

########################################################################################################################
# Set up the camera Matrix with the RGB Intrinsic Parameters:
camera_matrix_A = np.array([[fx_rgb, 0, cx_rgb],[0,fy_rgb, cy_rgb],[0, 0, 1]])
A = camera_matrix_A
########################################################################################################################

# RGB Distortion Parameters
k1_rgb = 2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

# Depth Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

# Rotation
R = -np.array([9.9997798940829263e-01,  5.0518419386157446e-03, 4.3011152014118693e-03,
               -5.0359919480810989e-03, 9.9998051861143999e-01, -3.6879781309514218e-03,
               -4.3196624923060242e-03, 3.6662365748484798e-03, 9.9998394948385538e-01])

R = R.reshape((3, 3))
R = np.linalg.inv(R.T)

R = np.eye(3, dtype=int)

# 3D Translation
t_x = 2.5031875059141302e-02
t_z = -2.9342312935846411e-04
t_y = 6.6238747008330102e-04

# Translation Matrix:
translation = np.array([t_x,t_y,t_z])



# Parameters for making depth absolute.
depth_param1 = 351.3
depth_param2 = 1092.5


def projection_3d_to_2d(points_3d):
    '''
    :param points_3d: open3d.cpu.pybind.geometry.PointCloud or numpy.array
    :param A: Camera Matrix (shape: 3x3)
    :param R: Rotation Matrix (shape: 3x3)
    :param translation: Translation Matrix (shape: 1x3)
    :return: u & v Pixel Coordinates of the points in 2D space
    '''
    # convert PointCloud object to numpy array
    # Note: Points are stored as a row vector in the form: [X, Y, Z]
    # Calculations according to formulas from OpenCV
    if type(points_3d) == o3d.cpu.pybind.geometry.PointCloud:
        points_3d_array = np.asarray(points_3d.points)
    else:
        if points_3d.shape[1] == 3:
            points_3d_array = points_3d
        else:
            points_3d_array = np.transpose(points_3d)

    # apply transformation
    points_3d_camera = np.matmul(points_3d_array, R) + translation
    points_2d_homogeneous = np.matmul(points_3d_camera, A.T)
    # Normalization_process
    points_2d_normalized = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    points_2d = points_2d_normalized

    return points_2d