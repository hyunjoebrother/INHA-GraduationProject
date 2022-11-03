#from turtle import color
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2 as cv
import time

def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi*-1
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz

def pcl(d,rgb):
    
    H, W = d.shape[:2]

    d= (d - d.min()) / (d.max() - d.min())
    d=(1/(d+0.1))

    xyz = np.multiply(d, get_uni_sphere_xyz(H, W))
    
    xyzrgb = np.concatenate([xyz, rgb / 255.], 2)
    xyzrgb = xyzrgb.reshape(-1, 6)

    pcd = o3d.geometry.PointCloud()
    print("Generating point cloud")
    
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])
    
    
    return pcd

if __name__ == "__main__":

    d=cv.imread('./mid-image/origin360_2_disp_Joint3D60_gray.png')
    rgb=cv.imread('./mid-image/origin360_2_resize.jpg')
    output_path = './'
    pcd = pcl(d, rgb)
    
    downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    downpcd.estimate_normals()
    o3d.visualization.draw_geometries([downpcd],point_show_normal=True) # normal 이후
    # o3d.visualization.draw_geometries([pcd]) # normal 이전
    o3d.io.write_point_cloud(output_path+"_pcd_test"+".ply", downpcd)






