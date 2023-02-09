import open3d as o3d

from src.pointcloud_utils import load_pc_from_pcd

pts = load_pc_from_pcd("/home/arvc/PointCloud/LiDARCameraCalibration/pointCloud/1674820178548972800.pcd")

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcl])