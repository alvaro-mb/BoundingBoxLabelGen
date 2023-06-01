import os.path
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from scipy.optimize import fsolve
from scipy import spatial
from scipy.spatial.transform import Rotation as R
import pyransac3d as pyrsc

from src.pointcloud_utils import load_pc, Visualizer
from src.image_utils import CamModel, Image


# Paths
# PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/robot2/"
# IMAGES_PATH = PATH + "camera/data/"
# POINTCLOUD_PATH = PATH + "lidar/data/"
PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/experiment/"
IMAGES_PATH = PATH + "images/"
POINTCLOUD_PATH = PATH + "pointclouds/"

imgs = sorted(glob(os.path.join(IMAGES_PATH, "*.png")), key=os.path.getmtime)
pcls = sorted(glob(os.path.join(POINTCLOUD_PATH, "*.ply")), key=os.path.getmtime)
model_file = "/home/arvc/PointCloud/LiDARCameraCalibration/calib_results.txt"
d_range = (0, 80)


def get_plane_points(points, i, radius):
    """ Find plane points from an initial seed selected on the screen with LassoSelector.

        :param points: array of lidar points
        :param i:      indexes of the plane points selected on the screen
        :param radius: radius used to find near points from the plane points

        :return: list of final point cloud plane points
    """
    # Find plane points from an initial seed
    points_kdtree = spatial.KDTree(points)
    indexes = []
    while True:
        idxs = points_kdtree.query_ball_point(points[i], radius)
        if len(idxs) == 0:
            break
        new_indexes = []
        for idx in idxs:
            new_indexes.extend(idx)
        new_indexes = list(dict.fromkeys(new_indexes))  # filter repeated points
        i = [e for e in new_indexes if e not in indexes]
        indexes.extend(i)

    return points[indexes]


def equations(xyz, *data):
    """ Define 6 distance equations between checkerboard corners, 4 line equations where the corners are found and
        plane equations for getting corners 3D coordinates.

        :param xyz:  array with 12 coordinates from the same dimension from each corner
        :param data: data with the following attributes:
                   p0: spherical unit coordinates from top left corner
                   p1: spherical unit coordinates from top right corner
                   p2: spherical unit coordinates from bottom left corner
                   p3: spherical unit coordinates from bottom right corner
                   plane: plane object with dimension data

        :return: array of functions
    """

    # Get point values and plane dimension parameters
    c, plane = data
    p0, p1, p2, p3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

    # CHECKERBOARD DIMENSIONS
    w, h, d = plane.width, plane.height, plane.diagonal

    x, y, z = xyz[:4], xyz[4:8], xyz[8:]

    # Line equations
    f1 = x[0]/p0[0] - y[0]/p0[1]
    f2 = x[0]/p0[0] - z[0]/p0[2]
    f3 = x[1]/p1[0] - y[1]/p1[1]
    f4 = x[1]/p1[0] - z[1]/p1[2]
    f5 = x[2]/p2[0] - y[2]/p2[1]
    f6 = x[2]/p2[0] - z[2]/p2[2]
    f7 = x[3]/p3[0] - y[3]/p3[1]
    f8 = x[3]/p3[0] - z[3]/p3[2]

    # Distance equations
    # f0 = (x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2 - w**2
    # f1 = (x[3] - x[2])**2 + (y[3] - y[2])**2 + (z[3] - z[2])**2 - w**2
    # f2 = (x[2] - x[0])**2 + (y[2] - y[0])**2 + (z[2] - z[0])**2 - h**2
    # f3 = (x[3] - x[1])**2 + (y[3] - y[1])**2 + (z[3] - z[1])**2 - h**2
    # f4 = (x[3] - x[0])**2 + (y[3] - y[0])**2 + (z[3] - z[0])**2 - d**2
    # f5 = (x[2] - x[1])**2 + (y[2] - y[1])**2 + (z[2] - z[1])**2 - d**2
    f9 = (x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2 - ((x[3] - x[2])**2 + (y[3] - y[2])**2 + (z[3] - z[2])**2)
    f10 = (x[2] - x[0])**2 + (y[2] - y[0])**2 + (z[2] - z[0])**2 - ((x[3] - x[1])**2 + (y[3] - y[1])**2 + (z[3] - z[1])**2)
    f11 = (x[3] - x[2])**2 + (y[3] - y[2])**2 + (z[3] - z[2])**2 - w**2

    # Plane restriction equation
    f12 = np.linalg.det(np.matrix([[x[1]-x[0], y[1]-y[0], z[1]-z[0]], [x[2]-x[0], y[2]-y[0], z[2]-z[0]], [x[3]-x[0], y[3]-y[0], z[3]-z[0]]]))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])


def bounding_edge_corners(points, rotations, plane):
    """ Get the bounding edge corners with minimum error.
        The rectangle with minimum mse that bounds the 2D perimeter points.
        :param: points:    array of projected plane points
        :param: rotations: Number of orientations in which 90º is divided
        :param: plane:     Plane object with the plane dimensions data

        :return: Base object with the [length, width], yaw, centroid [x,y] and fixed corner [x,y] from the base
    """

    # Get convex hull points only for x and y coordinates to get the perimeter
    hull_indexes = spatial.ConvexHull(points[:, :2]).vertices
    hull_points = points[hull_indexes, :2]
    perimeter = car2pol(hull_points)  # Perimeter in polar coordinates

    # Get the area of the rectangle for each orientation
    areas = []
    for i in range(0, rotations):
        car_perimeter = pol2car(perimeter)  # Cartesian perimeter
        # Calculate area of the rectangle
        area = (np.amax(car_perimeter[:, 0]) - np.amin(car_perimeter[:, 0])) * (np.amax(car_perimeter[:, 1]) - np.amin(car_perimeter[:,  1]))
        areas.append(area)
        perimeter[:, 1] = perimeter[:, 1] + (np.pi / 2) / rotations  # New yaw

    min_area_index = np.argmin(areas)
    yaw = min_area_index * (np.pi / 2) / rotations  # Yaw of the rectangle with minimum area
    perimeter[:, 1] = perimeter[:, 1] + yaw - np.pi / 2  # Rotate the perimeter to get size and position
    car_p = pol2car(perimeter)  # Cartesian perimeter

    # Get four corners from rectangle orientation
    car_corners = np.array([[np.amin(car_p[:, 0]), np.amax(car_p[:, 1])], [np.amax(car_p[:, 0]), np.amax(car_p[:, 1])],
                            [np.amin(car_p[:, 0]), np.amin(car_p[:, 1])], [np.amax(car_p[:, 0]), np.amin(car_p[:, 1])]])

    # Augment the rectangle to get the real plane size
    max_min = [np.amax(car_p[:, 0]) - np.amin(car_p[:, 0]), np.amax(car_p[:, 1]) - np.amin(car_p[:, 1])]
    width, height = plane.width, plane.height
    if width > height:
        w = np.max(max_min)
        h = np.min(max_min)
    else:
        w = np.min(max_min)
        h = np.max(max_min)
    car_corners[0] = [car_corners[0, 0] - (width - w)/2, car_corners[0, 1] + (height - h)/2]
    car_corners[1] = [car_corners[1, 0] + (width - w)/2, car_corners[1, 1] + (height - h)/2]
    car_corners[2] = [car_corners[2, 0] - (width - w)/2, car_corners[2, 1] - (height - h)/2]
    car_corners[3] = [car_corners[3, 0] + (width - w)/2, car_corners[3, 1] - (height - h)/2]
    pol_corners = car2pol(car_corners)
    pol_corners[:, 1] = pol_corners[:, 1] - yaw  # rotate back to the true position
    corners = pol2car(pol_corners)

    # Get centroid from the corners' mean
    centroid = [np.mean(corners[:, 0]), np.mean(corners[:, 1])]

    return centroid, corners


def pol2car(points):
    """ Converts 2D points in polar coordinates to cartesian coordinates.
        :param: points: Array of 2D points in polar coordinates

        :return: Array of 2D points in cartesian coordinates
    """
    rho = points.T[0]
    phi = points.T[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    cartesians = np.vstack([x, y]).T
    return cartesians

def car2pol(points):
    """ Converts 2D points in cartesian coordinates to polar coordinates.
        :param: points: Array of 2D points in cartesian coordinates

        :return: Array of 2D points in polar coordinates
    """
    x = points.T[0]
    y = points.T[1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    polars = np.vstack([rho, phi]).T
    return polars


def corner_finder(points, plane):
    """ Steps to find 4 plane corners given the hand-selected plane points and the plane dimensions.
            1. Find a plane equation with ransac and project points to founded plane.
            2. Find the center of the plane.
            3. Knowing the distance center-to-corner, trace a line from center to the farthest plane point
               and get two opposite corners
            4. Knowing the width and height of the plane, get the other two corners from all possible solutions.

        :param points: array of hand-selected plane points.
        :param plane:  Plane object containing dimension information of the plane

        :return: array of 4 corners founded.
    """

    # Get plane equation from using Ransac
    rsc_plane = pyrsc.Plane()
    plane_eq, _ = rsc_plane.fit(points, thresh=0.01, maxIteration=20000)
    A, B, C, D = plane_eq
    if C > 0:
        A, B, C, D = -A, -B, -C, -D
    normal = np.array([A, B, C])

    # Project plane points onto the plane
    proj_points = []
    for p in points:
        k = - (A*p[0] + B*p[1] + C*p[2] + D) / (A**2 + B**2 + C**2)
        pp = [k*A + p[0], k*B + p[1], k*C + p[2]]
        proj_points.append(pp)
    proj_points = np.array(proj_points)

    # Get the center of mass (fix as a plane center)
    center = []
    for i in range(proj_points.shape[1]):  # for x, y, z
        center.append(np.sum(proj_points[:, i]) / proj_points.shape[0])

    # Define v1 as a vector normal to x-axis and plane normal
    v1 = np.cross(np.array([1, 0, 0]), normal)

    # Transform proj_points to a new coordinate system with center as origin and normal as z-axis
    M = np.eye(4)
    M[:3, 0] = v1 / np.linalg.norm(v1)
    M[:3, 2] = normal / np.linalg.norm(normal)
    M[:3, 1] = np.cross(M[:3, 2], M[:3, 0])
    M[:3, 3] = np.array(center)
    # Add a fourth dimension to the points with value 1
    proj_points_ = np.vstack([proj_points.T, np.ones(proj_points.shape[0])])
    proj_points_ = np.dot(np.linalg.inv(M), proj_points_)[0:3, :].T

    # Get center and corners in new coordinate system
    center, corners = bounding_edge_corners(proj_points_, 360, plane)

    # Transform corners to original coordinate system adding a third dimension (z=0)
    corners = np.hstack([corners, np.zeros((corners.shape[0], 1))])
    # Add a fourth dimension to the points with value 1
    corners = np.vstack([corners.T, np.ones(corners.shape[0])])
    # Transform corners to original coordinate system
    corners = np.dot(M, corners)[0:3, :].T

    return corners


class Plane:
    """ Class for defining reference plane dimensions. """
    def __init__(self, width, height, diagonal=None):
        self.width = width
        self.height = height
        self.diagonal = diagonal
        if self.diagonal is None:
            self.diagonal = np.sqrt(self.width**2 +self.height**2)


def get_corners(points, image, camera_model, plane, lidar_plane = 0):
    """ Get the transformation matrix from the plane to the equirectangular image.
        :param points:       array of lidar 3D points
        :param image:        fisheye image
        :param camera_model: CameraModel object containing camera parameters
        :param plane:        Plane object containing dimension information of the plane
        :param lidar_plane:  1 for showing the lidar plane, 0 for not showing it

        :return: array of 3D camera corners
        :return: array of 3D lidar corners
        :return: PointCloud object
    """

    # Plot equirectangular image and point cloud
    vis = Visualizer(points, image)
    vis.reflectivity_filter(0.2)
    vis.get_spherical_coord(lidar2camera=0)
    vis.encode_values(d_range=d_range)
    equirect_lidar = Image(image=image, cam_model=camera_model, points_values=vis.pixel_values)
    equirect_lidar.sphere_coord = vis.spherical_coord
    equirect_lidar.lidar_projection(pixel_points=0)

    # Select plane points from plotted lidar
    fig, ax = plt.subplots(1)
    ax.imshow(equirect_lidar.eqr_image)
    pts = ax.scatter(x=equirect_lidar.eqr_coord[0], y=equirect_lidar.eqr_coord[1], c=equirect_lidar.points_values,
                     s=0.05, cmap=vis.cmap)
    plane_point = np.round(plt.ginput(timeout=0))[0]
    pixel_window = 2
    plane_index_points = np.where((equirect_lidar.eqr_coord[0] > plane_point[0] - pixel_window ) &
                                  (equirect_lidar.eqr_coord[0] < plane_point[0] + pixel_window ) &
                                  (equirect_lidar.eqr_coord[1] > plane_point[1] - pixel_window ) &
                                  (equirect_lidar.eqr_coord[1] < plane_point[1] + pixel_window ))[0]
    plt.close(fig)

    # Find plane points from an initial seed
    plane_points = get_plane_points(vis.lidar3d, plane_index_points, radius=0.04)

    # Find corners from plane points founded with corner_finder
    lidar_corners3d_ = np.asarray(corner_finder(plane_points, plane))

    # Reorder corners
    m = lidar_corners3d_[:, 1] + lidar_corners3d_[:, 2]
    lidar_corners3d = np.zeros((4, 3))
    lidar_corners3d[0, :] = lidar_corners3d_[np.argmax(m)]
    lidar_corners3d[3, :] = lidar_corners3d_[np.argmin(m)]
    lidar_corners3d_del = np.delete(lidar_corners3d_, [np.argmin(m), np.argmax(m)], axis=0)
    lidar_corners3d[1, :] = lidar_corners3d_del[np.argmin(lidar_corners3d_del[:, 1])]
    lidar_corners3d[2, :] = lidar_corners3d_del[np.argmax(lidar_corners3d_del[:, 1])]

    if lidar_plane == 1:
        # plot 3D points. Plot plane_points and lidar_corners3d with different colors. Plot lines between corners
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], s=0.5, c="#2d40ff")
        ax.scatter(lidar_corners3d[:, 0], lidar_corners3d[:, 1], lidar_corners3d[:, 2], s=5, c="#fa2a00")
        ax.plot([lidar_corners3d[0, 0], lidar_corners3d[2, 0], lidar_corners3d[3, 0], lidar_corners3d[1, 0],
                 lidar_corners3d[0, 0]],
                [lidar_corners3d[0, 1], lidar_corners3d[2, 1], lidar_corners3d[3, 1], lidar_corners3d[1, 1],
                 lidar_corners3d[0, 1]],
                [lidar_corners3d[0, 2], lidar_corners3d[2, 2], lidar_corners3d[3, 2], lidar_corners3d[1, 2],
                 lidar_corners3d[0, 2]], c="#fa2a00")
        ax.set_xlabel('X Label'), ax.set_ylabel('Y Label'), ax.set_zlabel('Z Label'), ax.axis('equal')
        plt.show()

    # # Plot equirectangular image and get 2D calibration pattern corners manually
    # equirect_image = Image(image, cam_model=camera_model)
    # equirect_image.fish2equirect()
    # plt.imshow(equirect_image.eqr_image)
    # print("Zoom in and select 4 checkerboard corners: top left, top right, bottom left, bottom right")
    # plt.title("Zoom in and select 4 checkerboard corners: top left, top right, bottom left, bottom right")
    # image2d_points = plt.ginput(5)
    # image2d_points = np.round(image2d_points[1:]).astype(int)
    # plt.close()
    # # Extract unit sphere corners coordinates from equirectangular image
    # equirect_image.eqr_coord = image2d_points.T
    # equirect_image.image2norm()
    # equirect_image.equirect2sphere()
    # image3d_sphere = equirect_image.sphere_coord

    # Plot fisheye image and get 2D calibration pattern corners manually
    fish_image = Image(image, cam_model=camera_model)
    plt.imshow(fish_image.image)
    print("Zoom in and select 4 checkerboard corners: top left, top right, bottom left, bottom right")
    plt.title("Zoom in and select 4 checkerboard corners: top left, top right, bottom left, bottom right")
    image2d_points = plt.ginput(5, timeout=0)
    image2d_points = np.round(image2d_points[1:]).astype(int)
    plt.close()
    # Extract unit sphere corners coordinates from fisheye image
    fish_image.spherical_proj = np.flip(image2d_points.T, axis=0)
    fish_image.fisheye2sphere()
    image3d_sphere = fish_image.sphere_coord

    # Get 3D corners coordinates from camera reference system
    # Change initial solutions if needed, remain x positive to get the plane in front of the camera
    xyz0 = np.array(np.reshape(lidar_corners3d.T, 12))
    xyz = fsolve(equations, xyz0, args=(image3d_sphere, plane))
    camera_corners3d = np.reshape(xyz, (3, 4)).T

    return camera_corners3d, lidar_corners3d, vis

def get_rotation_and_translation(camera_corners3d, lidar_corners3d, pointcloud, show=0, camera_model=None):
    """
    Estimate rotation and translation between camera and lidar reference systems
        :param camera_corners3d: array of 3D camera corners coordinates
        :param lidar_corners3d:  array of 3D lidar corners coordinates
        :param pointcloud:       PointCloud object
        :param show:             1 to show the lidar onto equirectangular projection,
                                 2 to show the lidar onto the fisheye image
                                 0 for not showing anything
        :param camera_model:     CameraModel object containing camera parameters

        :return: array of euler angles of rotation
        :return: array of translation values
        :return: error between transformed lidar corners and camera corners
    """

    # Estimate transform matrix between corners
    # print('\n3D camera corners coordinates: \n', camera_corners3d,
    #       '\n\n 3D lidar corners coordinates: \n', lidar_corners3d)
    pointcloud.estimate_transform_matrix(camera_corners3d, lidar_corners3d)
    print('\nTransformation matrix: \n', pointcloud.transform_matrix)
    # print lidar_corners3d transformed to camera reference system
    lidar_corners3d_transformed = np.matmul(pointcloud.transform_matrix, np.vstack((lidar_corners3d.T, np.ones((1, lidar_corners3d.shape[0]))))).T[:, :3]
    # print('\nLidar corners coordinates transformed to camera reference system: \n', lidar_corners3d_transformed)
    # Get error between transformed lidar corners and camera corners
    err = np.linalg.norm(lidar_corners3d_transformed - camera_corners3d, axis=1)
    mean_err = np.mean(err)
    std_err = np.std(err)

    euler = R.from_matrix(pointcloud.transform_matrix[:3, :3]).as_euler('xyz', degrees=True)
    # print('\nEuler angles from rotation matrix in degrees: \n', euler)
    # print('\nTranslation vector in meters: \n', pointcloud.transform_matrix[:3, 3])

    # assert that camera model is not None if show is not 0
    assert (show == 0) or (camera_model is not None), "Camera model is None"
    if show != 0:
        # Transform lidar points to camera reference system
        pointcloud.lidar_corners = lidar_corners3d
        pointcloud.camera_corners = camera_corners3d
        pointcloud.lidar_onto_image(cam_model=camera_model, fisheye= show - 1)
        plt.show()

    return euler, pointcloud.transform_matrix[:3, 3], mean_err, std_err


if __name__ == "__main__":

    # Big plane parameters
    width1, height1 = 1.89, 1.706
    big_plane = Plane(width1, height1)

    # Small plane parameters
    width2, height2 = 0.594, 0.412
    small_plane = Plane(width2, height2)

    # Define camera model from calibration file
    cam_model = CamModel(model_file)

    rotations = []
    translations = []
    kabsch_errors = []
    kabsch_std = []

    for i, p in zip(imgs, pcls):

        # Read image and pointcloud
        image = mpimg.imread(i)
        points = load_pc(p)

        # Get corners coordinates twice, one for each plane to get more points from the 3D space
        camera_corners1, lidar_corners1, pc1 = get_corners(points, image, cam_model, big_plane, lidar_plane=0)
        camera_corners2, lidar_corners2, pc2 = get_corners(points, image, cam_model, small_plane, lidar_plane=0)
        camera_corners = np.vstack((camera_corners1, camera_corners2))
        lidar_corners = np.vstack((lidar_corners1, lidar_corners2))

        rotation, translation, mean_error, std_error = get_rotation_and_translation(camera_corners, lidar_corners, pc1, show=0, camera_model=cam_model)
        kabsch_errors.append(mean_error)
        kabsch_std.append(std_error)

        rotations.append(rotation)
        translations.append(translation)

    kabsch_mean = np.mean(kabsch_errors)

    # Print kabsch errors and standard deviation
    print('\nMean Kabsch error: ', kabsch_errors)
    print('\nKabsch error standard deviation: ', kabsch_std)

    # Plot kabsch errors with standard deviation bars and a line for the mean
    plt.figure()
    plt.errorbar(np.arange(len(kabsch_errors)), kabsch_errors, yerr=kabsch_std, fmt='o', label='Kabsch error')
    plt.plot(np.arange(len(kabsch_errors)), np.ones(len(kabsch_errors)) * kabsch_mean, label='Mean Kabsch error')
    plt.xlabel('Image number')
    plt.ylabel('Error (m)')
    plt.title('Kabsch error')
    plt.legend()
    plt.show()

    # Get the mean of the rotations and translations
    rotations = np.array(rotations)
    translations = np.array(translations)
    mean_rotation = np.mean(rotations, axis=0)
    mean_translation = np.mean(translations, axis=0)

    # Get the error of the rotations and translations
    rotations_error = rotations - np.repeat(mean_rotation, rotations.shape[0], axis=0).reshape(rotations.shape[1], rotations.shape[0]).T
    translations_error = translations - np.repeat(mean_translation, translations.shape[0], axis=0).reshape(rotations.shape[1], rotations.shape[0]).T

    # Print rotations and translations errors
    print("\nRotations: ", rotations)
    print("\nTranslations: ", translations)
    print("\nMean rotation: ", mean_rotation)
    print("\nMean translation: ", mean_translation)
    print('\nRotations errors: ', rotations_error)
    print('\nTranslations errors: ', translations_error)
    print("\nMean rotation error: ", np.std(rotations, axis=0))
    print("\nMean translation error: ", np.std(translations, axis=0))

    # Get the mean of the error of the rotations and translations
    mean_rotations_error = np.mean(abs(rotations_error))
    mean_translation_error = np.mean(translations_error)

    # Plot rotation errors bars
    plt.figure()
    plt.bar(np.arange(len(rotations_error)) - 0.3, abs(rotations_error[:, 0]), 0.3, label='Rotation error x axis')
    plt.plot(np.arange(len(rotations_error)) - 0.3, np.ones(len(rotations_error)) * abs(mean_rotations_error[0]),
             label='Mean rotation error x axis')
    plt.bar(np.arange(len(rotations_error)), abs(rotations_error[:, 1]), 0.3, label='Rotation error y axis')
    plt.plot(np.arange(len(rotations_error)), np.ones(len(rotations_error)) * abs(mean_rotations_error[1]),
             label='Mean rotation error y axis')
    plt.bar(np.arange(len(rotations_error)) + 0.3, abs(rotations_error[:, 2]), 0.3, label='Rotation error z axis')
    plt.plot(np.arange(len(rotations_error)) + 0.3, np.ones(len(rotations_error)) * abs(mean_rotations_error[2]),
             label='Mean rotation error z axis')
    plt.xlabel('Image number')
    plt.ylabel('Error (degrees)')
    plt.title('Rotation error')
    plt.legend()
    plt.show()
    # Plot translation errors bars
    plt.figure()
    plt.bar(np.arange(len(translations_error)), np.linalg.norm(translations_error, axis=1), label='Translation error')
    plt.plot(np.arange(len(translations_error)),
             np.ones(len(translations_error)) * np.linalg.norm(mean_translation_error), label='Mean translation error',
             color='red')
    plt.xlabel('Image number')
    plt.ylabel('Error (m)')
    plt.title('Translation error')
    plt.legend()
    plt.show()

    # Print the results
    print("\nMean rotation error: ", mean_rotations_error)
    print("\nMean translation error: ", mean_translation_error)