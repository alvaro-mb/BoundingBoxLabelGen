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
imgs = sorted(glob(os.path.join('RosbagOMNI', '*.png')))

# Inference
results = model(imgs)

# Results
results.show()  # or .save()

print(results.pandas().xyxy[0])
