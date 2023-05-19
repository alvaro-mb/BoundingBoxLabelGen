from glob import glob
import os
import shutil
from src.labeler import labeler
from src.json_writters import *


if __name__ == "__main__":
    # Read images and point clouds from the dataset
    DATA_PATH = "path"
    POINTCLOUD_PATH = DATA_PATH + "/pcd"
    IMAGES_PATH = DATA_PATH + "/images"
    pcls = sorted(glob(os.path.join(POINTCLOUD_PATH, "*.pcd")), key=os.path.getmtime)
    imgs = sorted(glob(os.path.join(IMAGES_PATH, "*.png")), key=os.path.getmtime)

    assert imgs.__len__ == pcls.__len__, "There should be the same number of images and point clouds."

    directory = "directory"
    label_list = labeler(pcls, imgs, supervisely=1)  # Set supervisely = 1 to generate a supervisely labeled project
    proj = Project(label_list, directory)
    pcl_directory = directory + "ds0/pointcloud"
    if not os.path.exists(pcl_directory):
        os.makedirs(pcl_directory)
    shutil.copytree(POINTCLOUD_PATH, pcl_directory, dirs_exist_ok=True)