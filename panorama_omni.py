from glob import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.omni2panoramic import omni2panoramic

PANORAMA_PATH = '/home/arvc/PointCloud/rosbags/2022-12-07-15-22-43/robot0/camera/panoramas'
IMAGES_PATH = '/home/arvc/PointCloud/rosbags/2022-12-07-15-22-43/robot0/camera/data'
images = sorted(glob(os.path.join(IMAGES_PATH, "*.png")), key=os.path.getmtime)

x = 558
y = 756
out_r = 405  # External radius
in_r = 75  # Internal radius

if not os.path.exists(PANORAMA_PATH):
    os.makedirs(PANORAMA_PATH)

for img in images:
    image_array = mpimg.imread(img)
    panorama = omni2panoramic(image_array, x, y, out_r, in_r)
    # img_plot = plt.imshow(panorama)
    # plt.show()
    file_name = os.path.basename(img)
    mpimg.imsave(PANORAMA_PATH + '/' + file_name, panorama)

