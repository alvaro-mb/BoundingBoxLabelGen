import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.pointcloud_utils import load_pc, Visualizer
from src.image_utils import CamModel

# home_DIR = "/home/arvc/PointCloud/rosbags/2022-12-07-15-22-43/robot0"
# PCL_DIR = home_DIR + "/lidar/data/1670422966055838464.pcd"
# image_DIR = home_DIR + "/camera/panoramas/1670422966049728000.png"
# home_DIR = "/home/arvc/PointCloud/LiDARCameraCalibration"
# PCL_DIR = home_DIR + "/LiDAR/1677168911642341120.pcd"
# image_DIR = home_DIR + "/fisheye/1677168911633312000.png"
IMAGES_PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/robot0/camera/data/"
POINTCLOUD_PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/robot0/lidar/data/"

imgs = sorted(glob(os.path.join(IMAGES_PATH, "*.png")), key=os.path.getmtime)

model_file = "/home/arvc/PointCloud/LiDARCameraCalibration/calib_results.txt"

# VFOV = np.deg2rad(120)
# HFOV = np.deg2rad(200)
d_range = (0, 80)

dir_save = 'LidarImages'
if not os.path.exists(dir_save):
    os.mkdir(dir_save)
filename = 'Lidar_onto_camera.png'
saveto = dir_save + '/' + filename

image = mpimg.imread(imgs[15])
file_name = os.path.basename(imgs[15])
img_timestamp = int(file_name.split(".")[0])
pcl_timestamp = str(int(np.round((img_timestamp - 2500000000) / 100000000))) # 2.5 delay
pointcloud = glob(os.path.join(POINTCLOUD_PATH, pcl_timestamp + "*.ply"))
points = load_pc(pointcloud[0])

camera_model = CamModel(model_file)
pcl = Visualizer(points, image)  # , value='reflectivity')
pcl.reflectivity_filter()

pcl.lidar_onto_image(camera_model, d_range=d_range)

# pcl.lidar_to_panorama(projection_type='spherical', d_range=d_range, saveto=saveto)

# pcl.birds_eye_point_cloud(side_range=(-30, 30), fwd_range=(-30, 30), res=0.1, h_range=(-1.7, 3))

plt.show()

