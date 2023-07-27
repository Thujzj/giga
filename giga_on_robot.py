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

from vgn.detection_implicit import VGNImplicit
from vgn.experiments.clutter_removal import State
from vgn.perception import TSDFVolume, CameraIntrinsic
from vgn.utils.transform import Transform, Rotation
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.visual import grasp2mesh, plot_voxe_as_cloud, plot_tsdf_with_grasps

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

def image_pross(depth_img, rgb_img):
    
    # depth  = plt.imread("/mnt/data/optimal/jiazhongjie/github_codes/GIGA/scripts/image/depth.png")
    # depth_img = cv2.imread('scripts/image/depth_0.png', cv2.IMREAD_ANYDEPTH)
    # rgb_img = cv2.imread("scripts/image/color_0.png", cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    depth_img = depth_img/1000.0
    depth_img = depth_img.astype(np.float32)
    return depth_img, rgb_img


def get_object_in_camera(quat_ee_in_base, pos_ee_in_base, translation_base_in_table):
    # quat_ee_in_base = np.array([-0.9799, 0.1252, -0.1527, 0.0256])
    # pos_ee_in_base = np.array([0.494, -0.2197, 0.360])
    quat_camera_in_ee = np.array([-0.00106, -0.00108, 0.708838,0.705369,])
    pos_camera_in_ee = np.array([0.0514359,-0.0396951,-0.0430446])

    T_camera_in_ee = trans_matrix_from_quat_translation(quat_camera_in_ee,pos_camera_in_ee)
    T_ee_in_base = trans_matrix_from_quat_translation(quat_ee_in_base,pos_ee_in_base)

    T_camera_in_base = T_ee_in_base.dot(T_camera_in_ee)

    rotation_base_in_table = np.eye(3)
    # translation_base_in_table = np.array([-0.55,0.3,0.2])

    T_base_in_table = trans_matrix_from_rotation_translation(rotation_base_in_table,translation_base_in_table)
    T_camera_in_table = T_base_in_table.dot(T_camera_in_base)
    T_table_in_camera = np.linalg.inv(T_camera_in_table)
    return T_table_in_camera


def excute_grasp(grasp, table_in_base):
    # pos = grasp.pose.translation
    angle = grasp.pose.rotation.as_euler('xyz')
    if angle[2] > np.pi / 2 or angle[2] < -np.pi / 2:
        reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
        pose = grasp.pose * reflect
    else:
        pose = grasp.pose
    OBJECT_IN_TABLE = pose
    TABLE_IN_BASE =  Transform(Rotation.identity(), table_in_base)
    OBJECT_IN_BASE = TABLE_IN_BASE * OBJECT_IN_TABLE
    EE_IN_BASE = 0  #TODO
    delta_move = EE_IN_BASE - OBJECT_IN_BASE
    
    

def main(args):
    planner = VGNImplicit(args.model,
                        args.type,
                        best=True,
                        qual_th=0.8,
                        rviz=False,
                        force_detection=True,
                        out_th=0.1,
                        resolution=args.resolution)
    intrinsic = {
            "width": 640,
            "height": 480,
            "K": [596.382688, 0.0, 333.837001, 0.0, 596.701788, 254.401211, 0.0, 0.0, 1.0]
                }
    while True:
        TSDF = TSDFVolume(args.size, args.resolution)
        HIGH_RES_TSDF = TSDFVolume(args.size, 120, color_type='rgb')
        for _ in range(3):
            depth, rgb = 0, 0
            quat_ee_in_base, pos_ee_in_base, translation_base_in_table = 0,0,0

            depth_img, rgb_img = image_pross(depth, rgb)
            # rgb = plt.imread("scripts/image/color_0.png")
            extrinsic = get_object_in_camera(quat_ee_in_base, pos_ee_in_base, translation_base_in_table)
            # data = [[rgb_img, depth_img, intrinsic, extrinsics],]
            intrinsics = CameraIntrinsic.from_dict(intrinsic)
            extrinsics = Transform.from_matrix(extrinsic)
            TSDF.integrate(depth_img, intrinsics, extrinsics)
            HIGH_RES_TSDF.integrate(depth_img, intrinsics, extrinsics, rgb_img)
            
            
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(args.lower, args.upper)
        pc = HIGH_RES_TSDF.get_cloud()
        # pc = tsdf.get_cloud()
        pc = pc.crop(bounding_box)
        state = State(TSDF, pc)
        grasps, scores, _ = planner(state)
        if len(grasps) == 0:
            break
        elif len(np.where(scores > 0.8)) > 0:
            select =np.random.choice(np.where(scores > 0.8)) 
            excute_grasp(grasps[select])
        else:
            select = np.random.randint(0, len(grasps))
            excute_grasp(grasps[select])
            
            
   
    
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="/mnt/data/optimal/jiazhongjie/github_codes/GIGA/data/runs/23-07-10-09-40_dataset=data_packed_train_random_raw_4M_new,augment=False,net=giga,batch_size=32,lr=2e-04/best_vgn_giga_val_acc=0.9113.pt")
    parser.add_argument("--type", type=str, default="giga")
    parser.add_argument("--size", type=float, default=0.4)
    parser.add_argument("--resolution", type=float, default=40)
    parser.add_argument("--lower", type=float, nargs=3, default=[0.02, 0.02, 0.005])
    parser.add_argument("--upper", type=float, nargs=3, default=[0.28, 0.28, 0.3])
    args = parser.parse_args()
    main(args)
