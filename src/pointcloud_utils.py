import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpi
from sklearn import cluster


def load_pc_from_pcd(pcd_path):
    """ Load PointCloud data from pcd file. """
    p = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(p.points, dtype=np.float32)


def scale_to_255(a, mn, mx, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255.
        Optionally specify the data type of the output (default is uint8).
    """
    return (((a - mn) / float(mx - mn)) * 255).astype(dtype)


class PointCloud:
    """ Class for point clouds and its methods. """

    v_res = 0.3515625
    h_res = 0.3515625

    def __init__(self, lidar3d: np.ndarray, image: np.ndarray):
        """ Definition of point cloud attributes:
            :param lidar3d: Array of points in LiDAR coordinates (x, y, z)/(x, y, z, r)
            :param image:   Array of image points.
        """
        self.lidar3d = lidar3d
        self.image = image
        self.original_lidar3d = lidar3d
        self.reflectance = None
        self.depth = None
        self.coord_img = None

    def __len__(self):
        return self.lidar3d.shape[0]

    def set_reflectance(self):
        """ Set reflectance from lidar3d points. """
        assert self.lidar3d.shape[1] >= 4, "There is no reflectance data in point cloud data file."
        self.reflectance = self.lidar3d[:, 3]

    def transform_lidar_to_camera(self):
        """ Transforms the 3D LIDAR points to the camera 3D coordinates system. """
        # MATRIX EXTRACTED FROM LIDAR-CAMERA CALIBRATION PROCESS
        transform_matrix = np.array([[1, 0, 0, 0.18],
                                     [0, 1, 0, -0.055],
                                     [0, 0, 1, -0.52],
                                     [0, 0, 0, 1]])

        points = np.vstack([self.lidar3d.T, np.ones(self.lidar3d.shape[0])])
        self.lidar3d = np.dot(transform_matrix, points)[0:3, :].T

    def projection(self, lidar2camera=1, projection_type=''):
        """ Takes points in a 3D space from LIDAR data and projects them onto a 2D space.
            The (0, 0) coordinates are in the middle of the 2D projection.
            :param: projection_type: Choose between 'cylindrical' or 'spherical'

            :return: ndarray of x and y coordinates of each point on the plane, being 0 the middle
        """

        if lidar2camera == 1:
            # Transform to camera 3D coordinates
            self.transform_lidar_to_camera()

        x_lidar = self.lidar3d[:, 0]
        y_lidar = self.lidar3d[:, 1]
        z_lidar = self.lidar3d[:, 2]
        # Distance relative to origin when looked from top
        self.depth = np.sqrt(x_lidar ** 2 + y_lidar ** 2)

        # Only get points further than certain distance
        min_dist = 0.4
        x_lidar = x_lidar[self.depth > min_dist]
        y_lidar = y_lidar[self.depth > min_dist]
        z_lidar = z_lidar[self.depth > min_dist]
        self.lidar3d = self.lidar3d[self.depth > min_dist, :]
        self.depth = self.depth[self.depth > min_dist]

        # Convert to Radians
        v_res_rad = PointCloud.v_res * (np.pi / 180)
        h_res_rad = PointCloud.h_res * (np.pi / 180)

        # PROJECT INTO IMAGE COORDINATES
        if projection_type == 'cylindrical':
            x2d = np.arctan2(-y_lidar, x_lidar) / h_res_rad
            y2d = (z_lidar / self.depth) / v_res_rad
            # y2d = np.arctan2(z_lidar, self.depth) / v_res_rad
        else:
            x2d = np.arctan2(-y_lidar, x_lidar) / h_res_rad  # x2d = -np.arctan2(y_lidar, x_lidar) / h_res_rad
            y2d = np.arctan2(z_lidar, self.depth) / v_res_rad

        return np.array([x2d, y2d]).T

    def lidar_image_coordinates(self, panorama_fov, projection_type=''):
        """ Get LiDAR points pixel coordinates onto an image.
            :param: panorama_fov:    Panorama image field of view in radians
            :param: projection_type: Choose between 'cylindrical' or 'spherical'
        """

        proj2d = self.projection(projection_type=projection_type)

        # Get LiDAR vertical angles
        angles = proj2d[:, 1] * PointCloud.v_res * (np.pi/180)

        # Get the y points pixel value being 0 the middle of the image
        y_pixel_up_down = angles / panorama_fov * self.image.shape[0]

        # Make the top of the image the y = 0 value
        y = self.image.shape[0] / 2 - np.amax(y_pixel_up_down) + abs(y_pixel_up_down - np.amax(y_pixel_up_down))

        # Minimum x value known (panorama image)
        x_min = -360.0 / PointCloud.h_res / 2
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
    def __init__(self, lidar3d: np.ndarray, image: np.ndarray, value: str = 'depth', cmap = 'jet'):
        super().__init__(lidar3d, image)
        self.value = value
        self.cmap = cmap

    def lidar_to_panorama(self, projection_type='', d_range=None, saveto=None):
        """ Takes points in 3D space from LIDAR data and projects them to a 2D image and saves that image.
            :param: projection_type: Choose between 'cylindrical' or 'spherical'
            :param: d_range:         If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param: saveto:          If a string is provided, it saves the image as that filename given
        """

        # GET 2D PROJECTED POINTS
        points2d = self.projection(lidar2camera=0, projection_type=projection_type)

        # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
        x_img = points2d[:, 0]
        x_min = -360.0 / PointCloud.h_res / 2  # Theoretical min x value based on sensor specs
        x_img -= x_min  # Shift
        x_max = int(360.0 / PointCloud.h_res)  # Theoretical max x value after shifting

        y_img = points2d[:, 1]
        y_min = np.amin(y_img)  # min y value
        y_img -= y_min  # Shift
        y_max = int(np.amax(y_img))  # max x value

        # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
        if self.value == 'reflectance':
            self.set_reflectance()
            pixel_values = self.reflectance  # Reflectance
        elif self.value == 'height':
            pixel_values = self.lidar3d[:, 2]
        elif d_range is not None:
            pixel_values = -np.clip(self.depth, a_min=d_range[0], a_max=d_range[1])
        else:
            pixel_values = self.depth

        # CONVERT TO IMAGE ARRAY
        img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
        x, y = np.trunc(y_img).astype(np.int32), np.trunc(x_img).astype(np.int32)
        img[x, y] = scale_to_255(pixel_values, mn=np.amin(abs(pixel_values)), mx=np.amax(abs(pixel_values)))

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

    def lidar_onto_image(self, panorama_fov, projection_type='', d_range=None, saveto=None):
        """ Shows 3D LiDAR points onto its matched image obtained at the same of time. Optionally saves the result to specified filename.
            :param: panorama_fov:    Panorama image field of view in radians
            :param: projection_type: Choose between 'cylindrical' or 'spherical'
            :param: d_range:         If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param: saveto:          If a string is provided, it saves the image as this filename
        """

        self.lidar_image_coordinates(panorama_fov, projection_type)

        # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
        if self.value == 'reflectance':
            self.set_reflectance()
            pixel_values = self.reflectance  # Reflectance
        elif self.value == 'height':
            pixel_values = self.lidar3d[:, 2]
        elif d_range is not None:
            pixel_values = -np.clip(self.depth, a_min=d_range[0], a_max=d_range[1])
        else:
            pixel_values = self.depth

        # PLOT IMAGE AND POINTS IN IMAGE COORDINATES
        plt.imshow(self.image)
        plt.scatter(x=self.coord_img[:, 0], y=self.coord_img[:, 1], c=pixel_values, s=0.05, cmap=self.cmap)

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
        x_img = (-y_lidar[indices]/res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (x_lidar[indices]/res).astype(np.int32)   # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0, 0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img -= int(np.floor(side_range[0]/res))
        y_img -= int(np.floor(fwd_range[0]/res))

        # CLIP HEIGHT VALUES - to between min and max heights
        if h_range is not None:
            pixel_values = np.clip(z_lidar[indices], a_min=h_range[0], a_max=h_range[1])
        else:
            pixel_values = z_lidar[indices]

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values = scale_to_255(pixel_values, mn=np.amin(abs(pixel_values)), mx=np.amax(abs(pixel_values)))

        # FILL PIXEL VALUES IN IMAGE ARRAY
        x_max = int((side_range[1] - side_range[0])/res)
        y_max = int((fwd_range[1] - fwd_range[0])/res)
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
