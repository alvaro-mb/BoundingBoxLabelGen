import numpy as np
from typing import List

from src.pointcloud_utils import PointCloud


class BoundingBox:
    """ Class for bounding boxes and its methods. """

    # Minimum bounding box size depending on the class
    min_bbox_dim = [[3.80, 1.60, 1.30],  # car
                    [1.50, 0.45, 1.20],  # bicycle
                    [0.30, 0.45, 1.20]]  # pedestrian
    # Standard bounding box size depending on the class
    standard_bbox_dim = [[4.60, 2.00, 1.60],  # car
                         [1.60, 0.50, 1.50],  # bicycle
                         [0.40, 0.60, 1.70]]  # pedestrian

    def __init__(self, labeled_points: np.ndarray, cls, key_id: List = None):
        """ Definition of bounding box attributes
            :param labeled_points: Array of labeled points in spatial coordinates (x, y, z)
            :param cls:            Label class. 0: car, 1: bicycle, 2: pedestrian
            :param key_id:         Lists with the object and figures keys and ids
        """
        self.labeled_points = labeled_points
        self.cls = cls
        self.position = []
        self.dimension = []
        self.rotation = []
        self.perimeter = None
        self.min_bbox = []
        self.standard_bbox = []
        self.key_id = key_id  # List: [object key, figure key, object id, figure id]

        if self.key_id is not None:
            assert len(self.key_id) == 4

    def defined_dimensions(self):
        self.min_bbox = BoundingBox.min_bbox_dim[self.cls]
        self.standard_bbox = BoundingBox.standard_bbox_dim[self.cls]

    def bounding_box_label_generator(self, n_orientations):
        """ Extracts the bounding box that bounds the labeled points.
            Defines position, dimension and rotation of the bounding box.
            :param: n_orientations: Number of orientations in which 90º is divided
        """

        # Extract bounding box base
        self.defined_dimensions()
        base = self.bounding_base(n_orientations)

        # Resize bounding box base if it is not big enough
        bbox_min = np.array(self.min_bbox)[0:2]
        bbox_standard = np.array(self.standard_bbox)[0:2]
        if max(base.sizes) > np.amax(bbox_min):
            if min(base.sizes) < np.amin(bbox_min):
                base.sizes[np.argmin(base.sizes)] = np.amin(bbox_standard)
        elif max(base.sizes) > np.amin(bbox_min):
            if max(base.sizes) > np.amin(bbox_standard):
                base.sizes[np.argmax(base.sizes)] = np.amax(bbox_standard)
                base.sizes[np.argmin(base.sizes)] = np.amin(bbox_standard)
            else:
                base.sizes[np.argmax(base.sizes)] = np.amin(bbox_standard)
                base.sizes[np.argmin(base.sizes)] = np.amax(bbox_standard)
        else:
            base.sizes[np.argmax(base.sizes)] = np.amin(bbox_standard)
            base.sizes[np.argmin(base.sizes)] = np.amax(bbox_standard)

        self.dimension[0], self.dimension[1] = base.sizes[0], base.sizes[1]
        self.rotation = [0, 0, base.yaw]

        # Rotate centroid from the fixed corner
        pol_rot_centroid = [
            np.sqrt((base.corner[0] - base.centroid[0]) ** 2 + (base.corner[1] - base.centroid[1]) ** 2),
            np.arctan2((base.corner[1] - base.centroid[1]), (base.corner[0]) - base.centroid[0]) - base.yaw]
        car_rot_centroid = [base.corner + pol_rot_centroid[0] * np.cos(pol_rot_centroid[1]),
                            base.corner + pol_rot_centroid[0] * np.sin(pol_rot_centroid[1])]

        # Get rotated new centroid
        rotated_pos = []
        if 0 < pol_rot_centroid[1] < np.pi():
            rotated_pos[1] = base.sizes[1] / 2 + car_rot_centroid[1]
        else:
            rotated_pos[1] = base.sizes[1] / 2 - car_rot_centroid[1]
        if np.pi() / 2 < pol_rot_centroid[1] < 3 * np.pi() / 2:
            rotated_pos[0] = base.sizes[0] / 2 - car_rot_centroid[0]
        else:
            rotated_pos[0] = base.sizes[0] / 2 + car_rot_centroid[0]

        # Rotate new centroid back to the true position
        pol_position = [np.sqrt((base.corner[0] - rotated_pos[0]) ** 2 + (base.corner[1] - rotated_pos[1]) ** 2),
                        np.arctan2((base.corner[1] - rotated_pos[1]), (base.corner[0]) - rotated_pos[0]) - base.yaw]
        self.position[0] = base.corner + pol_position[0] * np.cos(pol_position[1])
        self.position[1] = base.corner + pol_position[0] * np.sin(pol_position[1])

        # Resize height in case of possible occlusions
        zmax_min = np.array([np.amax(self.labeled_points[2]), np.amin(self.labeled_points[2])])
        height = zmax_min[0] - zmax_min[1]
        zground = -0.75
        while height < self.min_bbox[2]:
            if zmax_min[1] > zground:
                zmax_min[1] = zground
                height = zmax_min[0] - zmax_min[1]
            else:
                zmax_min[0] = self.min_bbox[2] + zground
                height = zmax_min[0] - zmax_min[1]
        self.dimension[2] = zmax_min.mean()

    def bounding_base(self, rotations):
        """ Get the bounding base with minimum error. The base with minimum mse that bounds the 2D perimeter points.
            :param: rotations: Number of orientations in which 90º is divided

            :return: Base object with the [length, width], yaw, centroid [x,y] and fixed corner [x,y] from the base
        """

        while True:
            # Extract perimeter
            self.extract_perimeter()

            # Get the mse for each different orientation
            mse = self.mse_generator(rotations)
            min_mse_i = np.array(mse).argmin()  # Minimum MSE index
            yaw = - (min_mse_i * (np.pi / 2) / rotations)
            self.perimeter[1] = self.perimeter[1] - yaw  # Rotate the perimeter to get size and position
            car_p = self.pol2car(self.perimeter)  # Cartesian perimeter
            width = np.amax(car_p[0]) - np.amin(car_p[0])
            length = np.amax(car_p[1]) - np.amin(car_p[1])
            sizes = [width, length]

            # Base centroid
            c = [np.array([np.amax(car_p[0]), np.amin(car_p[0])]).mean(),
                 np.array([np.amax(car_p[1]), np.amin(car_p[1])]).mean()]  # Rotated centroid
            pol_centroid = [np.sqrt(c[0] ** 2 + c[1] ** 2),
                            np.arctan2(c[1], c[0]) + yaw]  # Rotate back to the true position
            centroid = [pol_centroid[0] * np.cos(pol_centroid[1]), pol_centroid[0] * np.sin(pol_centroid[1])]

            # Perimeter centroid to get fixed corner
            p_centroid = [car_p[0].mean(), car_p[1].mean()]  # Rotated perimeter centroid
            xmax_min = np.array([np.amax(car_p[0]), np.amin(car_p[0])])
            ymax_min = np.array([np.amax(car_p[1]), np.amin(car_p[1])])  # Maximum and minimum base values
            corner_dist_x = [abs(xmax_min[0] - p_centroid[0]),
                             abs(xmax_min[0] - p_centroid[0])]  # Distance to x max and min
            corner_dist_y = [abs(ymax_min[1] - p_centroid[1]),
                             abs(ymax_min[1] - p_centroid[1])]  # Distance to y max and min
            c = [xmax_min[np.argmin(corner_dist_x)], xmax_min[np.argmin(corner_dist_y)]]  # Fixed corner rotated
            pol_corner = [np.sqrt(c[0] ** 2 + c[1] ** 2), np.arctan2(c[1], c[0]) + yaw]  # Rotate back to the true position
            fixed_corner = [pol_corner[0] * np.cos(pol_corner[1]), pol_corner[0] * np.sin(pol_corner[1])]

            # Check that bounding area size is not too big. If it is, repeat clustering
            max_factor = 1.6
            if (np.dot(self.standard_bbox[0:1], max_factor) <= np.asarray(sizes)).any:
                # Maybe use outliers removal if the difference is not that big
                self.labeled_points = PointCloud.pcl_label_clustering(self.labeled_points)
            else: break

        return self.Base(sizes, yaw, centroid, fixed_corner)

    def extract_perimeter(self):
        """ Extracts the perimeter points of the labeled points projections in the x-y plane. """
        points = self.car2pol(self.labeled_points)
        i = np.argsort(points[1])
        points = points[i]  # Sorted points by angle
        phi, rho = points[0][0], points[0][1]
        perimeter = []
        for point in points:
            if point[1] == rho:  # Revisar que los angulos para los mismos puntos sean iguales (deberían)
                if point[0] < phi:
                    phi = point[0]
            else:
                rho = points[1]
                perimeter.append([rho, phi])
        self.perimeter = np.array(perimeter)

    def mse_generator(self, r):
        """ Generates MSE. Error based on distances between perimeter points and bounding base from the origin.
            :param: r: Number of times the base gets rotated (always the same amount of degrees) in 90º

            :return: mse: List with MSE for each different orientation
        """

        mse = []
        for i in range(0, r):
            errors = []
            car_perimeter = self.pol2car(self.perimeter)  # Cartesian perimeter
            # Maximum and minimum base values
            xmax_min = np.array([np.amax(car_perimeter[0]), np.amin(car_perimeter[0])])
            ymax_min = np.array([np.amax(car_perimeter[1]), np.amin(car_perimeter[1])])
            # Get the corner closest to the origin after the rotation
            corner = [xmax_min[np.argmin(np.abs(xmax_min))], ymax_min[np.argmin(np.abs(ymax_min))]]
            for pol_point, car_point in zip(self.perimeter, car_perimeter):
                # Missing coordinate from the intersection between corner's lines and
                # the line which goes through origin and the point
                x = corner[1] / np.tan(pol_point[1])
                y = corner[0] * np.tan(pol_point[0])
                # Distances between point and intersection
                dist1 = np.sqrt((car_point[0] - x) ** 2 + (car_point[1] - corner[1]) ** 2)
                dist2 = np.sqrt((car_point[1] - corner[0]) ** 2 + (car_point[0] - y) ** 2)
                e = np.min(np.array([dist1, dist2]))  # The minimum distance is the good one
                errors.append(e)
            ers = np.array(errors) ** 2  # Squared errors
            mse.append(ers.mean())  # MSE
            self.perimeter[1] = self.perimeter[1] + (np.pi / 2) / r  # New yaw

        return mse

    @staticmethod
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

    @staticmethod
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

    class Base:
        """ Base data needed to create final bounding box. """

        def __init__(self, sizes: List[float], yaw: float, centroid: List[float], corner: List[float]):
            """ Definition of a base object. Base which bounds the 2D perimeter points.
                :param sizes:    List with [width, length] of the base
                :param yaw:      Base orientation
                :param centroid: List with centroid 2D coordinates of the base [x, y]
                :param corner:   List with the fixed corner 2D coordinates of the base [x, y]
            """
            self.sizes = sizes
            self.yaw = yaw
            self.centroid = centroid
            self.corner = corner