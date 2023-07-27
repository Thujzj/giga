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
from vgn.detection_implicit import VGNImplicit
from vgn.experiments.clutter_removal import State
from vgn.perception import TSDFVolume, CameraIntrinsic
from vgn.utils.transform import Transform, Rotation
from vgn.utils.comm import receive_msg, send_msg
from vgn.utils.visual import grasp2mesh, plot_voxel_as_cloud, plot_tsdf_with_grasps

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
    r = R.from_quat(np.array([-0.999, 0.1252, -0.1527, 0.0256]))
    
    extrinsics = np.eye(4)
    extrinsics[0:3, 0:3] = r.as_matrix()
    print("r.as_matrix():::::::::\n",r.as_matrix())
    extrinsics[0:3, 3] = np.array([[0.494], [-0.2197],[0.360]]).reshape(3)
    E = Transform(r, np.r_[0.494, -0.2197,0.360])
    origin = Transform(Rotation.identity(), np.r_[-0.6, 0, 0.15])
    extrinsics = E * origin.inverse()
    extrinsics = extrinsics.as_matrix()
    
    # depth  = plt.imread("/mnt/data/optimal/jiazhongjie/github_codes/GIGA/scripts/image/depth.png")
    depth_img = cv2.imread('scripts/image/depth_0.png', cv2.IMREAD_ANYDEPTH)
    rgb_img = cv2.imread("scripts/image/color_0.png", cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    # print(depth_img)
    depth_img = depth_img/1000.0
    depth_img = depth_img.astype(np.float32)
    # print("depth range !!!!!!!!!!!!!!!",np.max(depth_img),  np.min(depth_img))
    # #initial 2, 0.1
    # far = 0.8
    # near = 0.2
    # depth_img = (
    #         1.0 * far * near / (far - (far - near) * depth_img)
    #     )
    
    # print("depth range !!!!!!!!!!!!!!!",np.max(depth_img),  np.min(depth_img))
    # depth_shift = 1000.0
    # x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))


    # uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    # uv_depth[:,:,0] = x
    # uv_depth[:,:,1] = y
    # uv_depth[:,:,2] = depth_img/depth_shift
    # uv_depth = np.reshape(uv_depth, [-1,3])
    # color = rgb_img.reshape(-1,3) / 255.0
    # color = color[np.where(uv_depth[:,2]!=0),:].squeeze()

    # uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()

    # fx = 603.309
    # fy = 602.827
    # cx = 326.608
    # cy = 245.279
    # n = uv_depth.shape[0]
    # points = np.ones((n,4))
    # X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx 
    # Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy 
    # points[:,0] = X
    # points[:,1] = Y
    # points[:,2] = uv_depth[:,2]

    # transform the point from camera coordinate to world coordinate

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
    data = [[rgb_img, depth_img, intrinsic, np.linalg.inv(T_camera_in_table)],]
    grasps, scores, geometries = predict_grasp(args, planner=planner, data=data)
    
    print(grasps[0].pose.to_dict())
    print(grasps[1].pose.to_dict())
    print(grasps[2].pose.to_dict())
    print(grasps[3].pose.to_dict())
    # print(scores[0])


def predict_grasp(args, planner, data):
    if len(data[0]) == 3:
        high_res_tsdf = TSDFVolume(args.size, 120)
    else:
        high_res_tsdf = TSDFVolume(args.size, 120, color_type='rgb')
    tsdf = TSDFVolume(args.size, args.resolution)
    
    for sample in data:
        if len(sample) == 3:
            depth, intrinsics, extrinsics = sample
            rgb = None
        else:
            rgb, depth, intrinsics, extrinsics = sample
        intrinsics = CameraIntrinsic.from_dict(intrinsics)
        # print(intrinsics.width, intrinsics.height)
        extrinsics = Transform.from_matrix(extrinsics)
        tsdf.integrate(depth, intrinsics, extrinsics)
        high_res_tsdf.integrate(depth, intrinsics, extrinsics, rgb_img=rgb)
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(args.lower, args.upper)
    pc = high_res_tsdf.get_cloud()
    # pc = tsdf.get_cloud()
    pc = pc.crop(bounding_box)
    
    # pc = o3d.io.read_point_cloud("points/demo_2.pcd")
    
    # pcd_arr = np.asarray(pc.points)
    # print(pcd_arr.shape)
    o3d.io.write_point_cloud("scripts/img/demo.pcd", pc)
    
    state = State(tsdf, pc)
    grasps, scores, _ = planner(state)
    # fig = plot_voxel_as_cloud(tsdf.get_grid()[0])
    # print("????")
    # fig.savefig('scripts/image/output.png')
    # print(len(grasps))
    print(np.count_nonzero(tsdf.get_grid()[0]) )
    if len(grasps) > 0:
        fig = plot_tsdf_with_grasps(tsdf.get_grid()[0], [grasps[1]])
        print(scores)
    else:
        fig = plot_voxel_as_cloud(tsdf.get_grid()[0])
    # fig.show()
    fig.savefig('scripts/img/output1_grasp.png')
    plt.close(fig)
    # while True:
    #     if plt.waitforbuttonpress():
    #         break
    # plt.close(fig)

    # grasp_meshes = [
    #     grasp2mesh(grasps[idx], 1).as_open3d for idx in range(len(grasps))
    # ]
    # geometries = [pc] + grasp_meshes

    # from copy import deepcopy
    # grasp_bck = deepcopy(grasps[0])
    # grasp_mesh_bck = grasp2mesh(grasp_bck, 1).as_open3d
    # grasp_mesh_bck.paint_uniform_color([0, 0.8, 0])

    # pos = grasps[0].pose.translation
    # # pos[2] += 0.05
    # angle = grasps[0].pose.rotation.as_euler('xyz')
    # print(pos, angle)
    # if angle[2] > np.pi / 2 or angle[2] < - np.pi / 2:
    #     reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
    #     grasps[0].pose = grasps[0].pose * reflect
    # pos = grasps[0].pose.translation
    # angle = grasps[0].pose.rotation.as_euler('xyz')
    # print(pos, angle)
    # # grasps[0].pose = Transform(Rotation.from_euler('xyz', (angle[0], angle[1], angle[2])), pos)
    # grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    # grasp_mesh.paint_uniform_color([0.8, 0, 0])
    # geometries = [high_res_tsdf.get_mesh(), grasp_mesh, grasp_mesh_bck]
    if len(grasps) == 0:
        return [], [], [high_res_tsdf.get_mesh()]
    pos = grasps[0].pose.translation
    # pos[2] += 0.05
    angle = grasps[0].pose.rotation.as_euler('xyz')
    print("pos and angle",pos, angle)
    if angle[2] > np.pi / 2 or angle[2] < - np.pi / 2:
        reflect = Transform(Rotation.from_euler('xyz', (0, 0, np.pi)), np.zeros((3)))
        grasps[0].pose = grasps[0].pose * reflect
    pos = grasps[0].pose.translation
    angle = grasps[0].pose.rotation.as_euler('xyz')
    print("pos, angle::::!!!!!!", pos, angle)
    # grasps[0].pose = Transform(Rotation.from_euler('xyz', (angle[0], angle[1], angle[2])), pos)
    grasp_mesh = grasp2mesh(grasps[0], 1).as_open3d
    grasp_mesh.paint_uniform_color([0, 0.8, 0])
    geometries = [high_res_tsdf.get_mesh(), grasp_mesh]
    #exit(0)
    
    return grasps, scores, geometries
    

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
