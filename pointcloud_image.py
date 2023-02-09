import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.pointcloud_utils import load_pc_from_pcd, Visualizer

home_DIR = "/home/arvc/PointCloud/rosbags/2022-12-07-15-22-43/robot0"
PCL_DIR = home_DIR + "/lidar/data/1670422966055838464.pcd"
image_DIR = home_DIR + "/camera/panoramas/1670422966049728000.png"
points = load_pc_from_pcd(PCL_DIR)
image = mpimg.imread(image_DIR)

panorama_FOV = np.deg2rad(120)
d_range = (0, 80)
projection = 'spherical'

dir_save = 'LidarImages'
if not os.path.exists(dir_save):
    os.mkdir(dir_save)
filename = 'image.png'
saveto = dir_save + '/' + filename

pcl = Visualizer(points, image)

# pcl.lidar_onto_image(panorama_FOV, 'cylindrical', d_range=d_range)

pcl.lidar_to_panorama(projection_type=projection, d_range=d_range, saveto=saveto)

# pcl.birds_eye_point_cloud(side_range=(-30, 30), fwd_range=(-30, 30), res=0.1, h_range=(-1.7, 3))

plt.show()

