

import numpy as np
import math
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
from rgbd_model import SingleObEncoder, DiffusionPolicy, EMAModel, RotationTransformer
# from torch_ema import ExponentialMovingAverage
import copy

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import open3d as o3d
# from dt_apriltags import Detector
import quaternion
from pytorch3d.transforms import quaternion_to_matrix
import transforms3d
import matplotlib.pyplot as plt


class LfD():
    def __init__(self):

        # normalize
        input_range = torch.load('input_range.pt')
      
        self.input_max = input_range[0,:]
        self.input_min = input_range[1,:]
        self.input_mean = input_range[2,:]

        self.config_file = "./config/grits.yaml"
        import yaml
        with open(self.config_file, 'r') as file:
            cfg = yaml.safe_load(file)


        obs_encoder = SingleObEncoder(cfg)
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        self.diffusion_model = DiffusionPolicy(
            cfg,
            obs_encoder,
            noise_scheduler
        )

        self.diffusion_model.to(cfg["training"]["device"])
   
    
        checkpoint = torch.load('policy_rgbd.pth', map_location="cuda:0")
  
        self.diffusion_model.load_state_dict(checkpoint['dp_state_dict'])
        
    
     
    
    def to_eular(self, pose):
        # ori [w,x,y,z]
        # orientation_list = [pose[4], pose[5], pose[6], pose[3]]
        (roll, pitch, yaw) = transforms3d.euler.quat2euler(pose[3:])
        return np.array([pose[0], pose[1], pose[2], roll, pitch, yaw])
    
    def to_qua(self, pose):
        # ori [roll,pitch,yaw]
        orientation_list = [pose[3], pose[4], pose[5]]
        q = quaternion_from_euler(orientation_list)
        return np.array([pose[0], pose[1], pose[2], q[0], q[1], q[2], q[3]])
    
    # visualize pcd
    def depth_image_to_point_cloud(
        self, rgb, depth, intrinsic_matrix, depth_scale=1, remove_outliers=True, z_threshold=None, mask=None, device="cuda:0"):
        # process input
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 255).to(device)
        depth = torch.from_numpy(depth.astype(np.float32)).to(device)
        intrinsic_matrix = torch.from_numpy(intrinsic_matrix.astype(np.float32)).to(device)
        
        # depth image to point cloud
        h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        x = x.float()
        y = y.float()

        ones = torch.ones((h, w), dtype=torch.float32)
        xy1s = torch.stack((x, y, ones), dim=2).view(w * h, 3).t()
        xy1s = xy1s.to(device)

        depth /= depth_scale
        points = torch.linalg.inv(intrinsic_matrix) @ xy1s
        points = torch.mul(depth.view(1, -1, w * h).expand(3, -1, -1), points.unsqueeze(1))
        points = points.squeeze().T

        colors = rgb.reshape(w * h, -1)
        
        # masks
        if mask is not None:
            mask = torch.from_numpy(mask).to(device)
            points = points[mask.reshape(-1), :]
            colors = colors[mask.reshape(-1), :]
        
        # remove far points
        if z_threshold is not None:
            valid = (points[:, 2] < z_threshold)
            points = points[valid]
            colors = colors[valid]

        # create o3d point cloud
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        scene_pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

        return np.asarray(scene_pcd.points), np.asarray(scene_pcd.colors), scene_pcd

    # capture apriltag and calculate goal pose for transfering and pouring
    def estimate_target_pose(self, rgb, depth, method='tag', vis=False):

        intrinsic = np.array([[606.2581787109375, 0.0, 322.64874267578125], [0.0, 606.0323486328125, 235.183349609375], [0.0, 0.0, 1.0]])
        extrinsic = np.load('dataset/cam2base.npy')

        if method=='tag':
            detector = Detector(
                families='tag36h11',
                nthreads=6,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0
            )
            rgb = rgb.astype(np.uint8, 'C')
            grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            camera_params = (intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
            tags = detector.detect(grey, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.02)
            assert len(tags) > 0, "No tag detected"

            tag65_pose = np.eye(4)
            tag85_pose = np.eye(4)
            for tag in tags:
                if tag.tag_id==65:
                    tag65_pose[0:3, 0:3] = tag.pose_R
                    tag65_pose[0:3, 3] = tag.pose_t.reshape(-1)
                elif tag.tag_id==85:
                    tag85_pose[0:3, 0:3] = tag.pose_R
                    tag85_pose[0:3, 3] = tag.pose_t.reshape(-1)

            # trasnfer tag pose to world coordinate
            tag65_pose = np.dot(extrinsic, tag65_pose)
            tag85_pose = np.dot(extrinsic, tag85_pose)

            # obtain food container's center
            food_center_pose = tag85_pose
            food_center_pose[0, 3] -= 0.12
            food_center_pose[1, 3] -= 0.02
            food_center_pose[2, 3] += 0.2

            # trasnfer to goal pose
            goal_pose = tag65_pose
            goal_pose[0, 3] -= 0.12  # 0.12
            goal_pose[1, 3] -= 0.07 # -0.02
            goal_pose[2, 3] += 0.2

        if vis:
            # visualize
            _, _, pcd = self.depth_image_to_point_cloud(rgb, depth, intrinsic, depth_scale=1000)
            pcd.transform(extrinsic)
            bowl = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            bowl.transform(food_center_pose)
            goal = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            goal.transform(goal_pose)
            o3d.visualization.draw_geometries([pcd, goal])

        return food_center_pose[0:3, 3], goal_pose[0:3, 3]
    
    def _normalize(self, data, input_max, input_min, input_mean):

        ranges = input_max - input_min
        data_normalize = torch.zeros_like(data)
        for i in range(3):
            if ranges[i] < 1e-4:
                data_normalize[i] = data[i] - input_mean[i]
            else:
                data_normalize[i] = -1 + 2 * (data[i] - input_min[i]) / ranges[i]
        data_normalize[3:] = data[3:]
        return data_normalize
    

    

    
    # ------- #
    # predict #
    #-------- #

    def rgb_transform(self, img):
        return transforms.Compose([
            transforms.Resize([240, 320]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])(img)

    def depth_transform(self, img):
        return transforms.Compose([
            transforms.Resize([240, 320]),
            transforms.ToTensor(),
        ])(img)
    
    def run_model(self, image_list, depth_list, eepose_list, seg_pcd):
        
        rotation_transformer_forward = RotationTransformer('quaternion', 'rotation_6d')
        rotation_transformer_backward = RotationTransformer('rotation_6d', 'quaternion')

        policy = self.diffusion_model
        policy.eval()

        rgb_input = []
        depth_input = []
        ee_input = []

        start_guidance = False
        with torch.no_grad():
            step = 0
            for i in range(image_list.shape[0]):
               
                rgb_input.append(self.rgb_transform(Image.fromarray(image_list[i].astype('uint8'), 'RGB')))
                depth_PIL = depth_list[i].astype(np.float32)
                depth_input.append(self.depth_transform(Image.fromarray(depth_PIL / np.max(depth_PIL))))
                

            rgb_in = torch.stack(rgb_input, dim=0) # [5, 3, 240, 320]
            depth_in = torch.stack(depth_input, dim=0)  # [5, 1, 240, 320]
            
            obs_in = torch.unsqueeze(torch.cat((rgb_in, depth_in), 1), 0).to('cuda:0', dtype=torch.float32) # [1, 5, 4, 240, 320]
        
            # for i in range(len(depth_in)):
            #     print
            #     plt.imshow(depth_in[i].reshape(240, 320, 1))
            #     plt.show()
            
        
            action, spillage_prob = policy.predict_action((obs_in, eepose_list, seg_pcd))  

            # transform rotate
            action_publish = action.cpu().detach().numpy().squeeze(0) # [8, 9]
            action_position = action_publish[:, 0:3]
            action_rotation = rotation_transformer_backward.forward(action_publish[:, 3:])
            action_publish = np.concatenate((action_position, action_rotation), -1) # [8, 7]
            

            euler_action = []
            for action in action_publish:
                euler = self.to_eular(action)
                euler_action.append(euler)
                # print(action, euler)
               
            euler_action = np.array(euler_action)

        return euler_action, spillage_prob
              


