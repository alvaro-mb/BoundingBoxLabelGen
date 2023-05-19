import torch
import numpy as np

from src.pointcloud_utils import load_pc, PointCloud
from src.bbox_utils import BoundingBox
from src.json_writters import hex_gen, dec_gen


def labeler(pointclouds, images, supervisely = None):
    """ Gets the 3D bounding boxes from each pointcloud with image data.

    :param: pointclouds: Array of points with LiDAR coordinates (x, y, z)/(x, y, z, r)
    :param: images:      Array of panoramic images matched with the point clouds
    :param: supervisely: None default.
                         Set to 1 if you want the label class to save the keys and ids from objects and figures which
                         will lately be used to generate the supervisely project.

    :return: list of lists of labels of each point cloud
    """

    # Yolov5 model from pytorch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

    v_fov = np.deg2rad(120)
    h_fov = np.deg2rad(360)
    key_id = [hex_gen, hex_gen, dec_gen(7), dec_gen(9)]
    dslabel = []

    for img, pcl in zip(images, pointclouds):
        points = load_pc(pcl)  # Load point cloud
        pcl = PointCloud(points, img)  # Load points and image using PointCloud class

        # Transform 2D LiDAR coordinates to 2D images pixel coordinates
        pcl.lidar_image_coordinates(v_fov, h_fov, 'spherical')

        # Extracts 2D labels from Yolov5 using the loaded image
        image_labels = model(img)  # Yolov5 inference
        point_marks, classes = pcl.point_marker(image_labels)
        n_orientations = 90
        labels = []

        # Extract and save bounding box data from labeled points
        for marks, cls in zip(point_marks, classes):
            labeled_points = pcl.pcl_label_clustering(marks)  # Filter points which are not part of the detection
            if supervisely is None:
                bbox = BoundingBox(labeled_points, cls)
                labels.append(bbox.bounding_box_label_generator(n_orientations))
            else:
                bbox = BoundingBox(labeled_points, cls, key_id)
                labels.append(bbox.bounding_box_label_generator(n_orientations))
                key_id[0], key_id[1] = hex_gen, hex_gen
                key_id[2], key_id[3] = key_id[2] + 1, key_id[3] + 1

        dslabel.append(labels)

    return dslabel
