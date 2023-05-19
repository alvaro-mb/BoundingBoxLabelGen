import torch
import os
from glob import glob

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Images
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
# imgs = sorted(glob(os.path.join('Kitti', '*.png')))
# imgs = 'PanoramicViews/Street-View.png'
# imgs = '/home/arvc/PointCloud/rosbags/2022-12-07-15-22-43/robot0/camera/panoramas/1670423807186023000.png'
PATH = "/home/arvc/PointCloud/LiDARCameraCalibration/fisheyecalibration/robot0/camera/data"
# PATH = "/home/arvc/Escritorio/1679312143858256000r.png"
imgs = sorted(glob(os.path.join(PATH, '*.png')))

# Inference
results = model(imgs[15])  # [imgs[0], imgs[5], imgs[10], imgs[15], imgs[20]])

# Results
results.show()  # or .save()

print(results.pandas().xyxy[0])
