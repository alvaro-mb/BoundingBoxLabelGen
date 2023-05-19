from glob import glob
import os
import cv2
import numpy as np
import matplotlib.image as mpi
import matplotlib.pyplot as plt
from src.image_utils import CamModel, Image


FISHEYE_PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/fisheyecalibration/robot0/camera/data"
EQUIRECT_PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/fisheyecalibration/robot0/camera/spherical_projection"
model_file = "/home/arvc/PointCloud/LiDARCameraCalibration/calib_results.txt"

imgs = sorted(glob(os.path.join(FISHEYE_PATH, "*.png")), key=os.path.getmtime)

s = np.zeros([1080, 1920])

for img in imgs:
    camera_model = CamModel(model_file)
    image = Image(img, camera_model)
    # gray = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
    # s = np.add(s, gray)
    file_name = os.path.basename(img)
    image.fish2equirect()
    # plt.imshow(image.eqr_image)
    # plt.show()
    # mpi.imsave(EQUIRECT_PATH + '/' + file_name, image.eqr_image)

# mean = s / np.size(imgs)
# histogram, bin_edges = np.histogram(mean, bins=256, range=(0.0, 1.0))
# plt.plot(bin_edges[0:-1], histogram)
# plt.show()
