import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
sys.path.append("C:\\Users\\jiazj20\\Desktop\\data\\src")
from vgn.perception import TSDFVolume, CameraIntrinsic
from vgn.utils.transform import Transform


def trans_matrix_from_quat_translation(quat,translation):
    pose = np.zeros((4,4))
    pose[:3,:3] = R.from_quat(quat).as_matrix()
    pose[:3,3] = translation
    pose[3,3] = 1
    return pose

def trans_matrix_from_rotation_translation(rotation,translation):
    pose = np.zeros((4,4))
    pose[:3,:3] = rotation
    pose[:3,3] = translation
    pose[3,3] = 1
    return pose


def convert_voxel_to_cloud(voxel, size):

    assert len(voxel.shape) == 3
    lx, ly, lz = voxel.shape
    lx = lx / voxel.shape[0] * size[0]
    ly = ly / voxel.shape[1] * size[1]
    lz = lz / voxel.shape[2] * size[2]
    points = []
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if voxel[x, y, z] > 0:
                    points.append([
                        x / voxel.shape[0] * size[0],
                        y / voxel.shape[1] * size[1],
                        z / voxel.shape[2] * size[2], voxel[x, y, z]
                    ])

    return np.array(points)


def plot_voxel_as_cloud(voxel,
                        axis=None,
                        figsize=(5, 5),
                        marker='s',
                        s=8,
                        alpha=.8,
                        lim=[0.3, 0.3, 0.3],
                        elev=10,
                        azim=240,
                        fig=None,
                        *args,
                        **kwargs):
    cloud = convert_voxel_to_cloud(voxel, lim)
    points = cloud[:, :3]
    val = cloud[:, 3]
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis

    # color = cm.coolwarm(val)
    sc = ax.scatter(*points.T,
                    marker=marker,
                    s=s,
                    alpha=alpha,
                    c=val,
                    cmap=plt.cm.get_cmap('RdYlBu_r'),
                    *args,
                    **kwargs)
    plt.colorbar(sc, ax=ax)
    ax.set_xlim3d(0, lim[0])
    ax.set_ylim3d(0, lim[1])
    ax.set_zlim3d(0, lim[2])

    return fig

    
def main():

    intrinsic = {
            "width": 640,
            "height": 480,
            "K": [596.382688, 0.0, 333.837001, 0.0, 596.701788, 254.401211, 0.0, 0.0, 1.0]
                }

    

    
    depth_img = cv2.imread('scripts/image/depth_0.png', cv2.IMREAD_ANYDEPTH)
    rgb_img = cv2.imread("scripts/image/color_0.png", cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    depth_img = depth_img/1000.0
    depth_img = depth_img.astype(np.float32)

    quat_ee_in_base = np.array([-0.9799, 0.1252, -0.1527, 0.0256])
    pos_ee_in_base = np.array([0.494, -0.2197, 0.360])

    quat_camera_in_ee = np.array([-0.00106, -0.00108, 0.708838,0.705369,])
    pos_camera_in_ee = np.array([0.0514359,-0.0396951,-0.0430446])

    T_camera_in_ee = trans_matrix_from_quat_translation(quat_camera_in_ee,pos_camera_in_ee)
    T_ee_in_base = trans_matrix_from_quat_translation(quat_ee_in_base,pos_ee_in_base)

    T_camera_in_base = T_ee_in_base.dot(T_camera_in_ee)

    rotation_base_in_table = np.eye(3)
    translation_base_in_table = np.array([-0.55,0.3,0.2])

    T_base_in_table = trans_matrix_from_rotation_translation(rotation_base_in_table,translation_base_in_table)

    T_camera_in_table = T_base_in_table.dot(T_camera_in_base)
    print("T_camera_in_table\n",T_camera_in_table)
    # rgb = plt.imread("scripts/image/color_0.png")
    # data = [[rgb_img, depth_img, intrinsic, np.linalg.inv(T_camera_in_table)],]
    TSDF = TSDFVolume(0.5, 120, "rgb")
    intrinsics = CameraIntrinsic.from_dict(intrinsic)
    extrinsics = Transform.from_matrix(T_camera_in_table)
    TSDF.integrate(depth_img, intrinsics, extrinsics.inverse(), rgb_img)
    pc = TSDF.get_cloud()
    bounding_box = o3d.geometry.AxisAlignedBoundingBox([0.02, 0.02, 0.005], [0.28, 0.28, 0.3])
    o3d.visualization.draw_geometries([pc])
    pc = pc.crop(bounding_box)
    print(np.count_nonzero(pc))
    o3d.io.write_point_cloud("scripts/img/demo.pcd", pc)
    # o3d.visualization.draw([pc])
    
    
    mesh = TSDF._volume.extract_triangle_mesh()
    tsdf = TSDF._volume.extract_volume_tsdf()
    
    # print(tsdf)
    # o3d.visualization.draw_geometries([tsdf])
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    
    
main()
    
    
    