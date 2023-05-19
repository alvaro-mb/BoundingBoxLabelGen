import numpy as np
import os
import cv2
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt
import matplotlib.image as mpi
from sklearn import cluster

from src.image_utils import Image


def load_pc(path):
    """ Load PointCloud data from pcd or ply file. """
    extension = os.path.splitext(path)[1]
    if extension == ".pcd":
        p = o3d.io.read_point_cloud(path)
        return np.asarray(p.points, dtype=np.float32)
    if extension == ".ply":
        p = PlyData.read(path)
        return np.asarray((p.elements[0].data['x'], p.elements[0].data['y'], p.elements[0].data['z'], p.elements[0].data['reflectivity'])).T


def scale_to_255(a, mn, mx, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255.
        Optionally specify the data type of the output (default is uint8).
    """
    return (((a - mn) / float(mx - mn)) * 255).astype(dtype)


class PointCloud:
    """ Class for point clouds and its methods. """

    v_res = np.deg2rad(0.3515625)
    h_res = np.deg2rad(0.3515625)

    def __init__(self, lidar3d, image):
        """ Definition of point cloud attributes:
            :param lidar3d: Directory or array of points in LiDAR coordinates (x, y, z)/(x, y, z, r)
            :param image:   Directory or array of image points.
        """

        assert isinstance(lidar3d, str) or isinstance(lidar3d, np.ndarray), "lidar3d must be a directory pcd string or a matrix array. "
        if isinstance(lidar3d, str):
            self.lidar3d = load_pc(lidar3d)
            self.original_lidar3d = load_pc(lidar3d)
        else:
            self.lidar3d = lidar3d
            self.original_lidar3d = lidar3d

        assert isinstance(image, str) or isinstance(image, np.ndarray), "image must be a directory image string or a matrix array. "
        if isinstance(image, str):
            self.image = mpi.imread(image)
        else:
            self.image = image

        # Distance relative to origin when looked from top
        self.depth = np.sqrt(self.lidar3d[:, 0] ** 2 + self.lidar3d[:, 1] ** 2)
        self.original_depth = np.sqrt(self.original_lidar3d[:, 0] ** 2 + self.original_lidar3d[:, 1] ** 2)

        self.spherical_coord = None
        self.coord_img = None
        if self.lidar3d.shape[1] >= 4:
            self.reflectivity = self.lidar3d[:, 3]
            self.original_reflectivity = self.original_lidar3d[:, 3]
            self.lidar3d = self.lidar3d[:, :3]
            self.original_lidar3d = self.original_lidar3d[:, :3]
        else:
            self.reflectivity = None
            self.original_reflectivity = None
        self.transform_matrix = np.identity(4)
        self.transform_matrix[2, 3] = -0.03618  # OUSTER TRANSFORM CORRECTION

    def __len__(self):
        return self.lidar3d.shape[0]

    def estimate_transform_matrix(self, lidar_pts, camera_pts):
        """ Estimate transform matrix between LiDAR and camera given 4

            :param lidar_pts:  array of 3D points from LiDAR coordinates
            :param camera_pts: array of 3D points from camera coordinates
        """

        assert lidar_pts.shape[0] == camera_pts.shape[0], 'There must be the same number of points.'

        N = lidar_pts.shape[0]  # total points

        centroid_A = np.mean(lidar_pts, axis=0)
        centroid_B = np.mean(camera_pts, axis=0)

        # center the points
        AA = lidar_pts - np.tile(centroid_A, (N, 1))
        BB = camera_pts - np.tile(centroid_B, (N, 1))

        # Get rotation matrix from svd
        H = np.dot(np.transpose(BB), AA)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = np.array(-R * np.matrix(centroid_B).T + np.matrix(centroid_A).T)

        self.transform_matrix = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])

    def define_transform_matrix(self):
        """ Define transformation matrix manually. """

        # MATRIX EXTRACTED FROM LIDAR-CAMERA CALIBRATION PROCESS
        t = [0, 0, 0]  # distances from camera to LiDAR
        r = [0, np.deg2rad(0), np.deg2rad(0)]
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(r[0]), -np.sin(r[0]), 0],
                       [0, np.sin(r[0]), np.cos(r[0]), 0],
                       [0, 0, 0, 1]])

        Ry = np.array([[np.cos(r[1]), 0, np.sin(r[1]), 0],
                       [0, 1, 0, 0],
                       [-np.sin(r[1]), 0, np.cos(r[1]), 0],
                       [0, 0, 0, 1]])

        Rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0, 0],
                       [np.sin(r[2]), np.cos(r[2]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        self.transform_matrix = np.dot(np.dot(Rz, Ry), Rx)
        self.transform_matrix[0][3] = t[0]
        self.transform_matrix[1][3] = t[1]
        self.transform_matrix[2][3] = t[2]
        # self.transform_matrix = [[-7.78983183e-04, -6.92083970e-02,  1.40196939e-01,  1.62669288e+00],
        #                          [-2.14531542e-01, -1.82117729e-02, -1.80218643e-02,  3.91008408e-01],
        #                          [-6.20808498e-03, -3.26799219e-01,  6.58804986e-01, -6.61006656e-02],
        #                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

    def transform_lidar_to_camera(self):
        """ Transforms the 3D LIDAR points to the camera 3D coordinates system. """

        points = np.vstack([self.lidar3d.T, np.ones(self.lidar3d.shape[0])])
        self.lidar3d = np.vstack(np.dot(self.transform_matrix, points)[0:3, :].T)
        self.depth = np.sqrt(self.lidar3d[:, 0] ** 2 + self.lidar3d[:, 1] ** 2)

    def distance_filter(self):
        """ Filter points by distance. """
        min_dist = 0.4
        self.lidar3d = self.lidar3d[self.depth > min_dist, :]
        if self.reflectivity is not None:
            self.reflectivity = self.reflectivity[self.depth > min_dist]
        self.depth = self.depth[self.depth > min_dist]

    def reflectivity_filter(self, percentage=0.05):
        """ Filter points by reflectivity. """
        min_reflectivity = percentage * 255
        self.lidar3d = self.lidar3d[self.reflectivity > min_reflectivity, :]
        self.depth = self.depth[self.reflectivity > min_reflectivity]
        self.reflectivity = self.reflectivity[self.reflectivity > min_reflectivity]

    def unfilter_lidar(self):
        self.lidar3d = self.original_lidar3d
        self.depth = self.original_depth
        self.reflectivity = self.original_reflectivity

    def projection(self, lidar2camera=1, projection_type=''):
        """ Takes points in a 3D space from LIDAR data and projects them onto a 2D space.
            The (0, 0) coordinates are in the middle of the 2D projection.
            :param: lidar2camera: flag to transform LiDAR points to camera reference system.
            :param: projection_type: Choose between 'cylindrical' or 'spherical'

            :return: ndarray of x and y coordinates of each point on the plane, being 0 the middle
        """

        if lidar2camera == 1:
            # Transform to camera 3D coordinates
            self.transform_lidar_to_camera()

        self.distance_filter()

        x_lidar = self.lidar3d[:, 0]
        y_lidar = self.lidar3d[:, 1]
        z_lidar = self.lidar3d[:, 2]

        # PROJECT INTO IMAGE COORDINATES
        if projection_type == 'cylindrical':
            x2d = np.arctan2(-y_lidar, x_lidar) / PointCloud.h_res
            y2d = (z_lidar / self.depth) / PointCloud.v_res
            # y2d = np.arctan2(z_lidar, self.depth) / v_res_rad
        else:  # spherical
            x2d = np.arctan2(-y_lidar, x_lidar) / PointCloud.h_res  # x2d = -np.arctan2(y_lidar, x_lidar) / h_res_rad
            y2d = np.arctan2(z_lidar, self.depth) / PointCloud.v_res

        return np.array([x2d, y2d]).T

    def get_spherical_coord(self, lidar2camera=1):
        """ Get coordinates from 3D LiDAR points onto the unit sphere.
            :param: lidar2camera: flag to transform LiDAR points to camera reference system.
        """

        if lidar2camera == 1:
            # Transform to camera 3D coordinates
            self.transform_lidar_to_camera()

        self.distance_filter()

        x_lidar = self.lidar3d[:, 0]
        y_lidar = self.lidar3d[:, 1]
        z_lidar = self.lidar3d[:, 2]

        # Project onto unit sphere
        n = np.linalg.norm(self.lidar3d, axis=1)
        x = x_lidar / n
        y = y_lidar / n
        z = z_lidar / n

        self.spherical_coord = np.vstack((x, y, z))

    def lidar_image_coordinates(self, v_fov, projection_type=''):
        """ Get LiDAR points pixel coordinates onto an image.
            :param: v_fov:           Vertical image field of view in radians
            :param: projection_type: Choose between 'cylindrical' or 'spherical'
        """

        proj2d = self.projection(projection_type=projection_type)

        # Get LiDAR vertical angles
        angles = proj2d[:, 1] * PointCloud.v_res

        # Get the y points pixel value being 0 the middle of the image
        y_pixel_up_down = angles / v_fov * self.image.shape[0]

        # Make the top of the image the y = 0 value
        y = self.image.shape[0] / 2 - np.amax(y_pixel_up_down) + abs(y_pixel_up_down - np.amax(y_pixel_up_down))

        # Minimum x value known (panorama image)
        x_min = -np.pi / PointCloud.h_res
        proj2d[:, 0] -= x_min

        # Scale the x value to get the x pixel coordinates
        x = self.image.shape[1] / np.amax(-x_min * 2) * proj2d[:, 0]

        # Array of point cloud points in pixel coordinates (x, y)
        self.coord_img = np.array([x, y]).T

    def point_marker(self, ilabels):
        """ Group points from point clouds in labels extracted from matched images with yolov5x.
            :param: ilabels: Torch tensor with image label data

            :return: Array of arrays of labeled points with LiDAR coordinates (x, y, z, r)
            :return: Array with label class data
        """

        label3d = []
        classes = []
        for ilabel in ilabels.xyxy[0]:
            marked_points = []
            if ilabel[5] <= 2:
                classes.append(ilabel[5])
                for ipoint, point in zip(self.coord_img, self.original_lidar3d):
                    if float(ilabel[0]) < ipoint[0] < float(ilabel[2]) and float(ilabel[1]) < ipoint[1] < float(ilabel[3]):
                        marked_points.append(point)
                label3d.append(marked_points)
        return np.array(label3d), np.array(classes)

    @staticmethod
    def pcl_label_clustering(label):
        """ Filters first point cloud labels.

                :param: label: Array of points labeled by yolov5x with LiDAR coordinates (x, y, z, r)

                :return: Array of final labeled points with spatial coordinates (x, y, z)
            """

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(label)

        ############################################################
        # REVISAR DISEÑO DEL ALGORITMO A IMPLEMENTAR Y HACER PRUEBAS
        ############################################################
        clustering = cluster.spectral_clustering(n_clusters=2, random_state=0,
                                                 assign_labels='cluster_qr').fit_predict(label[:, 0:2])
        groups = clustering[np.newaxis].T
        label_groups = np.hstack([label[:, 0:2], groups])

        # Find the biggest cluster
        group = np.bincount(groups).argmax()
        label_points = label_groups[label_groups[3, :] == group, :]

        ### VISUALIZE LABELED POINTS
        pcl_colors = np.zeros([label.shape[0], 3])
        pcl_colors[group] = [0, 1, 1]
        pc.colors = o3d.utility.Vector3dVector(pcl_colors)
        o3d.visualization.draw_geometries(pc)

        return label_points[:, 0:2]


class Visualizer(PointCloud):
    """ PointCloud visualizer methods. """
    def __init__(self, lidar3d: np.ndarray, image: np.ndarray, value = 'depth', cmap = 'jet'):
        super().__init__(lidar3d, image)
        self.value = value
        self.cmap = cmap
        self.pixel_values = None
        self.lidar_corners = None
        self.camera_corners = None

    def encode_values(self, d_range=None):
        """ What data to use to encode the value for each pixel.
            :param d_range: If tuple is provided, it is used for clipping distance values to be within a min and max range
        """

        if self.value == 'reflectivity':
            assert self.reflectivity is not None, "There is no reflectivity data in point cloud data file."
            self.pixel_values = self.reflectivity  # Reflectivity
        elif self.value == 'height':
            self.pixel_values = self.lidar3d[:, 2]
        elif d_range is not None:
            self.pixel_values = -np.clip(self.depth, a_min=d_range[0], a_max=d_range[1])
        else:
            self.pixel_values = self.depth

    def lidar_to_panorama(self, projection_type='', d_range=None, saveto=None):
        """ Takes points in 3D space from LIDAR data and projects them to a 2D image and saves that image.
            :param: projection_type: Choose between 'cylindrical' or 'spherical'
            :param: d_range:         If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param: saveto:          If a string is provided, it saves the image as that filename given
        """

        # GET 2D PROJECTED POINTS
        points2d = self.projection(lidar2camera=1, projection_type=projection_type)

        # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
        x_img = points2d[:, 0]
        x_min = -2 * np.pi / PointCloud.h_res / 2  # Theoretical min x value based on sensor specs
        x_img -= x_min  # Shift
        x_max = int(2 * np.pi / PointCloud.h_res)  # Theoretical max x value after shifting

        y_img = points2d[:, 1]
        y_min = np.amin(y_img)  # min y value
        y_img -= y_min  # Shift
        y_max = int(np.amax(y_img))  # max x value

        self.encode_values(d_range=d_range)

        # CONVERT TO IMAGE ARRAY
        img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
        x, y = np.trunc(y_img).astype(np.int32), np.trunc(x_img).astype(np.int32)
        img[x, y] = scale_to_255(self.pixel_values, mn=np.amin(abs(self.pixel_values)), mx=np.amax(abs(self.pixel_values)))

        # PLOT THE IMAGE
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=self.cmap)
        ax.xaxis.set_visible(False)  # Do not draw axis tick marks
        ax.yaxis.set_visible(False)  # Do not draw axis tick marks
        plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
        plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV

        # SAVE THE IMAGE
        if saveto is not None:
            img = np.flip(img, 0)
            mpi.imsave(saveto, img, cmap=self.cmap)

    def lidar_onto_image(self, cam_model=None, fisheye=0, d_range=None, saveto=None):
        """ Shows 3D LiDAR points onto its matched image obtained at the same of time. Optionally saves the result to specified filename.
            :param cam_model: Fisheye camera model loaded from calib.txt
            :param fisheye:   Project LiDAR onto fisheye image converted to spherical projection if 0
                              Project LiDAR onto fisheye image if 1.
            :param d_range:   If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param saveto:    If a string is provided, it saves the image as this filename
        """

        self.unfilter_lidar()
        self.get_spherical_coord()
        self.encode_values(d_range=d_range)

        if fisheye == 0:
            # GET LIDAR PROJECTED ONTO SPHERICAL PROJECTION FROM FISHEYE IMAGE
            lidar_proj = Image(image=self.image, cam_model=cam_model, points_values=self.pixel_values)
            lidar_proj.fish2equirect()
            lidar_proj.sphere_coord = self.spherical_coord
            lidar_proj.lidar_projection(pixel_points=0)
            plt.imshow(lidar_proj.eqr_image)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.scatter(x=lidar_proj.eqr_coord[0], y=lidar_proj.eqr_coord[1], c=lidar_proj.points_values, s=0.05, cmap=self.cmap)

            # Plot lidar and camera corners
            if self.lidar_corners is not None:
                lcorners = PointCloud(self.lidar_corners, self.image)
                lcorners.transform_matrix = self.transform_matrix
                lcorners.get_spherical_coord()
                lidar_proj.sphere_coord = lcorners.spherical_coord
                lidar_proj.lidar_projection(pixel_points=0)
                plt.scatter(x=lidar_proj.eqr_coord[0], y=lidar_proj.eqr_coord[1], c='r', s=0.5)
                # plt.plot([lidar_proj.eqr_coord[0, 0], lidar_proj.eqr_coord[0, 2], lidar_proj.eqr_coord[0, 3], lidar_proj.eqr_coord[0, 1], lidar_proj.eqr_coord[0, 0]],
                #          [lidar_proj.eqr_coord[1, 0], lidar_proj.eqr_coord[1, 2], lidar_proj.eqr_coord[1, 3], lidar_proj.eqr_coord[1, 1], lidar_proj.eqr_coord[1, 0]],
                #          'r', linewidth=1)
            if self.camera_corners is not None:
                ccorners = PointCloud(self.camera_corners, self.image)
                ccorners.get_spherical_coord(0)
                lidar_proj.sphere_coord = ccorners.spherical_coord
                lidar_proj.lidar_projection(pixel_points=0)
                plt.scatter(x=lidar_proj.eqr_coord[0], y=lidar_proj.eqr_coord[1], c='g', s=0.5)
                # plt.plot([lidar_proj.eqr_coord[0, 0], lidar_proj.eqr_coord[0, 2], lidar_proj.eqr_coord[0, 3], lidar_proj.eqr_coord[0, 1], lidar_proj.eqr_coord[0, 0]],
                #          [lidar_proj.eqr_coord[1, 0], lidar_proj.eqr_coord[1, 2], lidar_proj.eqr_coord[1, 3], lidar_proj.eqr_coord[1, 1], lidar_proj.eqr_coord[1, 0]],
                #          'g', linewidth=1)

        else:
            lidar_fish = Image(image=self.image, cam_model=cam_model, points_values=self.pixel_values)
            lidar_fish.sphere_coord = self.spherical_coord
            lidar_fish.change2camera_ref_system()
            lidar_fish.sphere2fisheye()
            lidar_fish.check_image_limits()
            plt.imshow(lidar_fish.image)
            u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
            plt.scatter(x=u, y=v, c=lidar_fish.points_values, s=0.05, cmap=self.cmap)

            # Plot lidar and camera corners
            if self.lidar_corners is not None:
                lcorners = PointCloud(self.lidar_corners, self.image)
                lcorners.transform_matrix = self.transform_matrix
                lcorners.get_spherical_coord()
                lidar_fish.sphere_coord = lcorners.spherical_coord
                lidar_fish.change2camera_ref_system()
                lidar_fish.sphere2fisheye()
                lidar_fish.check_image_limits()
                u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
                plt.scatter(x=u, y=v, c='r', s=0.5)
                # plt.plot([u[0], u[2], u[3], u[1], u[0]], [v[0], v[2], v[3], v[1], v[0]], 'r', linewidth=1)
            if self.camera_corners is not None:
                ccorners = PointCloud(self.camera_corners, self.image)
                ccorners.get_spherical_coord(0)
                lidar_fish.sphere_coord = ccorners.spherical_coord
                lidar_fish.change2camera_ref_system()
                lidar_fish.sphere2fisheye()
                lidar_fish.check_image_limits()
                u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
                plt.scatter(x=u, y=v, c='g', s=0.5)
                # plt.plot([u[0], u[2], u[3], u[1], u[0]], [v[0], v[2], v[3], v[1], v[0]], 'g', linewidth=1)

        if saveto is not None:
            plt.savefig(saveto, dpi=300, bbox_inches='tight')

    def lidar_onto_panorama(self, v_fov=0, h_fov=0, projection_type='', d_range=None, saveto=None):
        """ Shows 3D LiDAR points onto its matched image obtained at the same of time. Optionally saves the result to specified filename.
            :param v_fov:           Vertical image field of view in radians
            :param h_fov:           Horizontal image field of view in radians
            :param projection_type: Choose between 'cylindrical' or 'spherical'
            :param d_range:         If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param saveto:          If a string is provided, it saves the image as this filename
        """

        self.lidar_image_coordinates(v_fov, projection_type=projection_type)
        self.encode_values(d_range=d_range)

        # PLOT IMAGE AND POINTS IN IMAGE COORDINATES
        plt.imshow(self.image)
        x_img = self.image.shape[1] / 2
        index_pixels = ((x_img - h_fov * x_img / (2 * np.pi)) < self.coord_img[:, 0]) * (self.coord_img[:, 0] < (x_img + h_fov * x_img / (np.pi * 2)))
        self.pixel_values = self.pixel_values[index_pixels]
        self.coord_img = self.coord_img[index_pixels, :]
        plt.scatter(x=self.coord_img[:, 0], y=self.coord_img[:, 1], c=self.pixel_values, s=0.05, cmap=self.cmap)

        # SAVE THE IMAGE
        if saveto is not None:
            img = np.flip(self.image, 0)
            mpi.imsave(saveto, img, cmap=self.cmap)

    def birds_eye_point_cloud(self, side_range=(-10, 10), fwd_range=(-10, 10), res=0.1, h_range=None, saveto=None):
        """ Creates an 2D birds eye view representation of the point cloud. Optionally saves the image to specified filename.
            :param: side_range: (-left, right) in metres. Left and right limits of rectangle to look at
            :param: fwd_range:  (-behind, front) in metres. Back and front limits of rectangle to look at
            :param: res:        Desired resolution in metres to use.
                                Each output pixel will represent a square region res x res in size
            :param: h_range:    Used to truncate height values to minimum and maximum height relative to the sensor (metres)
            :param: saveto:     Filename to save the image.
        """

        x_lidar = self.lidar3d[:, 0]
        y_lidar = self.lidar3d[:, 1]
        z_lidar = self.lidar3d[:, 2]

        # INDICES FILTER - of values within the desired rectangle
        # Note left side is positive y axis in LIDAR coordinates
        ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
        ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
        indices = np.argwhere(np.logical_and(ff, ss)).flatten()

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-y_lidar[indices] / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (x_lidar[indices] / res).astype(np.int32)   # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0, 0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img -= int(np.floor(side_range[0] / res))
        y_img -= int(np.floor(fwd_range[0] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        if h_range is not None:
            pixel_values = np.clip(z_lidar[indices], a_min=h_range[0], a_max=h_range[1])
        else:
            pixel_values = z_lidar[indices]

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values = scale_to_255(pixel_values, mn=np.amin(abs(pixel_values)), mx=np.amax(abs(pixel_values)))

        # FILL PIXEL VALUES IN IMAGE ARRAY
        x_max = int((side_range[1] - side_range[0]) / res)
        y_max = int((fwd_range[1] - fwd_range[0]) / res)
        img = np.zeros([y_max, x_max], dtype=np.uint8)
        img[-y_img, x_img] = pixel_values  # -y because images start from top left

        # PLOT THE IMAGE
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=self.cmap)
        ax.xaxis.set_visible(False)  # Do not draw axis tick marks
        ax.yaxis.set_visible(False)  # Do not draw axis tick marks
        plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
        plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV

        # SAVE THE IMAGE
        if saveto is not None:
            mpi.imsave(saveto, img, cmap=self.cmap)
