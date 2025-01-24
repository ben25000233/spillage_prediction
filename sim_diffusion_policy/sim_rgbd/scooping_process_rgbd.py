from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import yaml
import torch
import numpy as np
import open3d as o3d
import random
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from BallGenerator import BallGenerator

from predict_rgbd import LfD
import torchvision.transforms as transforms
import cv2
import transforms3d
from dynamics_model.test_spillage import spillage_predictor
import json

class IsaacSim():
    def __init__(self):
        self.grain_type = "solid"
        self.lfd = LfD()
        self.spillage_predictor = spillage_predictor()
        self.default_height = 0.2
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"
        # initialize gym
        self.gym = gymapi.acquire_gym()
        #self.domainInfo = WeighingDomainInfo(domain_range=None, flag_list=None)

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "./urdf"
        self.gravity = -9.8
        self.create_sim()


        
        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_target = gymapi.Vec3(0.5, -0.18, 0.3)

        # Look at the first env
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, cam_target)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        
        self.All_poses = [torch.tensor([], device=self.device) for _ in range(self.num_envs)]
        
        
        self.All_steps = np.zeros(self.num_envs)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7]


    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()
        #args.physics_engine = gymapi.SIM_FLEX

        args.use_gpu = True
        args.use_gpu_pipeline = False
        self.device = 'cpu'
        self.num_envs = 1
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        
        sim_params.dt = 1.0/60
        sim_params.substeps = 10

 
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 4

        sim_params.physx.friction_offset_threshold = 0.00001  # lead 穿模
        sim_params.physx.friction_correlation_distance = 1
        # lead lag
        sim_params.physx.contact_offset = 0.0001
        
        sim_params.physx.rest_offset = 0.00001
        sim_params.physx.max_depenetration_velocity = 1

        
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
     

        # self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_table(self):

        # Load Bowl asset
        file_name = 'table/table.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        self.table_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
    def create_bowl(self):

        # Load Bowl asset
        file_name = 'bowl/bowl.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.fix_base_link = True
        asset_options.vhacd_params.resolution = 1500000
        self.bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.transparent_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, 'bowl/transparent_bowl.urdf', asset_options)


    def create_franka(self):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = "franka_description/robots/" + self.tool + "_franka.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 30000000
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file_franka, asset_options)
        self.franka_dof_names = self.gym.get_asset_dof_names(self.franka_asset)
        self.num_dofs += self.gym.get_asset_dof_count(self.franka_asset)

        self.hand_joint_index = self.gym.get_asset_joint_dict(self.franka_asset)["panda_hand_joint"]
        self.ee_handles = []
        # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][:].fill(3000.0)
        # self.franka_dof_props["armature"][:] = 100
      
        self.franka_dof_props["damping"][:].fill(500.0)
       
   
        self.franka_dof_props["effort"][:] = 500

        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        asset_options.tendon_limit_stiffness = 4000
        

    def add_franka(self):
        # create franka and set properties
        self.franka_handle = self.gym.create_actor(self.env_ptr, self.franka_asset, self.franka_start_pose, "franka", 0, 4, segmentationId = 1)
       
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.franka_handle)
        for k in range(11):
            body_shape_prop[k].thickness = 0.00001
            body_shape_prop[k].friction = 0.001
            body_shape_prop[k].rest_offset = 0.01

        
     
            
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.franka_handle, body_shape_prop)

        franka_sim_index = self.gym.get_actor_index(self.env_ptr, self.franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)

        self.franka_dof_index = [0,1,2,3,4,5,6,7,8]

        franka_hand = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.franka_handle, "panda_hand")
        self.ee_handles.append(franka_hand)
        self.franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(self.env_ptr, self.franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.gym.set_actor_dof_properties(self.env_ptr, self.franka_handle, self.franka_dof_props)

        
    def create_bolt(self):

        # Load bolt asset
        file_name = 'grains/bolt.urdf'
        asset_options = gymapi.AssetOptions()
        self.between_ball_space = 0.1
        asset_options.armature = 0.01
        asset_options.vhacd_params.resolution = 500000
        self.bolt_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        

    def create_ball(self):
       
        ball_size = "medium"

        if ball_size == "small" :
            self.ball_radius = 0.003
            self.ball_mass = 0.003
            # friction interval : 0 - 0.2
            self.ball_friction = 0.05
            self.ball_amount = 60

        elif ball_size == "medium" :
        
            self.ball_radius = 0.005
            self.ball_mass = 0.01
            # friction interval : 0 - 0.2
            self.ball_friction = 0.05
            self.ball_amount = 15
        
        elif ball_size == "large" :

            self.ball_radius = 0.02
            self.ball_mass = 0.03
            # friction interval : 0 - 0.2
            self.ball_friction = 0.05
            self.ball_amount = 1
        
             
    
        ballGenerator = BallGenerator()
        file_name = 'BallHLS.urdf'
        ballGenerator.generate(file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass, type = "solid")
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())

    def set_ball_property(self, ball_pose):
        
        ball_handle = self.gym.create_actor(self.env_ptr, self.ball_asset, ball_pose, "grain", 0, 0, segmentationId = 2)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, ball_handle)
  
        body_shape_prop[0].friction = self.ball_friction
        body_shape_prop[0].contact_offset = 0.001   # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0.000001      # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0.000001     # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 1       
  
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, ball_handle, body_shape_prop)
        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])
        color = gymapi.Vec3(1, 1, 1)
        self.gym.set_rigid_body_color(self.env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return ball_handle
    
        
    def add_solid(self):
        #add balls
        ball_pose = gymapi.Transform()
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle = []
        ball_amount = self.ball_amount
    
        z = 0.1 + self.ball_radius
        if self.ball_radius > 0.01 :
            ran = 2
        else :
            ran  = 6
    
        while ball_amount > 0:
            y = -0.18
            for j in range(ran):
                x = self.bowl_x - 0.04
                for k in range(ran):
                    ball_pose.p = gymapi.Vec3(x, y, z)
                    ball_handle = self.set_ball_property(ball_pose)
                    self.ball_handle.append(ball_handle)
                    x += self.ball_radius*2.2 
                y += self.ball_radius*2.2 
            z += self.ball_radius*2
            ball_amount -= 1




    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(0, 0.15 * spacing, spacing)
        self.create_bowl()
        self.create_ball()
        self.create_table()
        self.create_franka()
        self.create_bolt()
        
    
        # cache some common handles for later use
        self.camera_handles = [[]for _ in range(self.num_envs)]
        self.franka_indices = []
        self.envs = []
        self.ee_handles = []

        #set camera
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 320
        self.camera_props.height = 240
        self.camera_props.horizontal_fov = 57
        
        #store ball info
        self.spillage_amount = np.zeros(self.num_envs)
        self.pre_spillage = 0

        self.ball_handles = [[] for _ in range(self.num_envs)]
        self.spillage_amount = [[] for _ in range(self.num_envs)]

        self.record_ee_pose = []
        self.rgb_list = []
        self.depth_list = []
        self.seg_list = []
        self.seg_pcd_list = []
        self.binary_spillage = []
        self.total_spillage = 0
        self.init_spillage = 0

        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)

            # add bowl_1
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(1.8))
            # default : self.bowl_x = 0.51
            self.bowl_x = 0.56
            bowl_pose.p = gymapi.Vec3(self.bowl_x, -0.15 , 0.026) 
            

            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0, segmentationId = 4)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            # thickness(soft) = 0.0003, thickness(soft) = 0.007
            body_shape_prop[0].thickness = 0.0004      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            body_shape_prop[0].friction = 0.001
            body_shape_prop[0].rest_offset = 0.001
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl_1, body_shape_prop)
            
            
            #add bowl_2
            bowl_pose = gymapi.Transform()
            bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            bowl_pose.p = gymapi.Vec3(0.5, 0.15 , 0.02)   
            self.target_bowl_pose = np.array([bowl_pose.p.x, bowl_pose.p.y, bowl_pose.p.z])
            self.bowl_2 = self.gym.create_actor(self.env_ptr, self.transparent_bowl_asset, bowl_pose, "bowl_2", 0, 0)
            color = gymapi.Vec3(140/255, 227/255, 229/255)
            self.gym.set_rigid_body_color(self.env_ptr, self.bowl_2, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            # add tabel
            table_pose = gymapi.Transform()
            table_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.2)   
            self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0, segmentationId = 3)
            color = gymapi.Vec3(150/255, 150/255, 140/255)
            self.gym.set_rigid_body_color(self.env_ptr, self.table, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            

            
            #add ball

            self.add_solid()
            self.ball_handles[i] = self.ball_handle

            

            #add franka
            self.add_franka()

            #add camera_1
            camera_1 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_transform = gymapi.Transform()
            side_cam_pose = np.load('./../real_cam_pose/cam2base.npy')

            # 1.004056, -0.047730, 0.766758
            pose = side_cam_pose[:3, 3]

            camera_transform.p = self.cam_pos = gymapi.Vec3(pose[0], pose[1], pose[2])
            #assign
            z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(180))
            y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(55.5))
            x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(-4))

            camera_transform.r = z*y*x

        
            # self.cam_target = gymapi.Vec3(0.5, -0.18, 0.3)
            self.gym.set_camera_transform(camera_1, self.env_ptr, camera_transform)
          
            self.camera_handles[i].append(camera_1)

            '''
            # add camera_2(need modify)
            camera_2 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_offset = gymapi.Vec3(0.01, 0.0, 0.11)
            camera_rotation = gymapi.Quat(0.8, 0, 0.8, 0)
          
            self.gym.attach_camera_to_body(camera_2, self.env_ptr, self.ee_handles[i], gymapi.Transform(camera_offset, camera_rotation),
                                    gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_2)
            '''

        self.franka_actor_indices = to_torch(self.franka_indices, dtype = torch.int32, device = "cpu")


    def cal_spillage_scooped(self, reset = 0):
        # reset = 1 means record init spillage in experiment setting 
        current_spillage = 0
        spillage_amount = 0

        # calculate spillage and scoop amount
        for ball in self.ball_handles[0]:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[0], ball, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]
            y = body_states['pose']['p'][0][1]

            if z < 0:
                current_spillage += 1
        
        if reset == 1 : 
            self.init_spillage = current_spillage
        
        elif reset == 0:
            spillage_amount = current_spillage - self.pre_spillage
            if int(spillage_amount) == 0:
                self.binary_spillage.append(0)
            else :
                self.binary_spillage.append(1)
            # print(f"spillage : {spillage_amount}")
            
        elif reset == 2 :
            spillage_amount = current_spillage - self.init_spillage
            self.total_spillage = spillage_amount
        # print(f"spillage amount : {spillage_amount}")
        self.pre_spillage = int(current_spillage)

    
    def reset_franka(self):
        
        joint_list = np.load(f"down_trial.npy")
        # joint_list = np.load(f"joint_states.npy")
        init_joint = joint_list[10]
        
        # init_joint = [-1.60388427, 0.69085033, 1.37022825, -1.60295399, -0.69766643,  1.69337432, -0.99492682]
        action = init_joint
        action = np.append(action, 0.02)
        action = np.append(action, 0.02)
        franka_init_pose = torch.tensor(action, dtype=torch.float32)
       
        self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[:, self.franka_dof_index, 1] = 0
 

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices),
            len(self.franka_actor_indices)
        )


    def control_ik(self, dpose, damping=0.05):

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)

        return u

    def move_generate(self):
  
        if len(self.rgb_list) == 1:

            begin_rgb = self.rgb_list[0].copy()
      
            images = [begin_rgb] * 5
            depths = [self.depth_list[0].astype(np.float32)] * 5 
            eepose = [self.record_ee_pose[0].astype(np.float32)] * 5
            seg_pcd = [np.array(self.seg_pcd_list)[0]]*5

            
        else:
            images = self.rgb_list[-5:]  
            depths = self.depth_list[-5 :]
            eepose = self.record_ee_pose[-5:]
            seg_pcd = self.seg_pcd_list[-5:]


        image_array = np.array(images)
        depth_array = np.array(depths) 
        eepose_array = np.array(eepose)
        seg_pcd_array = np.array(seg_pcd)


        eeposes, spillage_prob = self.lfd.run_model(image_array, depth_array, eepose_array, seg_pcd_array)
  
        #offset to make spillage
        eeposes[:, 1] += 0.004

   
  
        return eeposes, spillage_prob
    

    

    def scooping_process(self):

        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)
        self.frame = 0
        dpose_index = 0
        total_action = []
        prob_list = [0]

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            # self.gym.render_all_camera_sensors(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            self.frame += 1
            if self.frame <= 60 : 
                if self.frame == 60 :
                    self.reset_franka()
                    self.cal_spillage_scooped(reset = 1) 

            else :   
        
                self.get_info()
                
                if dpose_index % 8 == 0 :
                    self.cal_spillage_scooped(reset = 0)
        
                    single_action, spillage_prob = self.move_generate()
                    total_action.extend(single_action)

                                
                    prob_list.append(spillage_prob.to("cpu").detach())
            
                ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
                init_eepose  = self.to_eular(np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z]))


                dpose = total_action[dpose_index] - init_eepose
                dpose[3] = -dpose[4]
                dpose[4:] = 0

                move = self.control_ik(dpose)

                self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_index, 0].squeeze(-1)[:, :7] + move
                
                dpose_index += 1
                self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()

                if dpose_index > 300 :
                    self.cal_spillage_scooped(reset = 2)
                    print(self.total_spillage)
                    # self.write_file()
                    break
                
                
            temp = self.dof_state[:, self.franka_dof_index, 0].clone()
            temp[:, -2:] = 0.02

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(temp),
                gymtorch.unwrap_tensor(self.franka_actor_indices),
                len(self.franka_actor_indices)
            )

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
       
        return np.array(prob_list)[:-1], np.array(self.binary_spillage)

    def to_eular(self, pose):

        orientation_list = [pose[3], pose[4], pose[5], pose[6]]
        (roll, pitch, yaw) = transforms3d.euler.quat2euler(orientation_list)
        return np.array([pose[0], pose[1], pose[2],roll,pitch, yaw ])
    

    def get_info(self, env_index = 0):

        # get camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
    
        # get eepose
        ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
        ee_pose_arr  = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z])
     
        self.record_ee_pose.append(ee_pose_arr)
        
        #get top_image

        top_color_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_COLOR)
        top_color_image = top_color_image.reshape(self.camera_props.height, self.camera_props.width, 4)
        self.rgb_list.append(top_color_image[ :, :, :3])


        # plt.imshow(top_color_image[30:135, 50:165, :3])
        # plt.show()
        # exit()
        
        
        top_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_DEPTH)
        self.depth_list.append(top_depth_image)
        nor_top_depth = self.normalize_image(top_depth_image[30:135, 50:165])
        
        #get segment pcd
        top_seg_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_SEGMENTATION)
        seg_pcd = self.depth_to_point_cloud(env_index, nor_top_depth,top_seg_image[30:135, 50:165], type = 0)
        self.seg_pcd_list.append(np.array(seg_pcd))
        
        
    def depth_to_point_cloud(self, env_index, depth_image, seg_image, type = 0):

        # Get camera intrinsics
        vinv = np.linalg.inv(np.array(self.gym.get_camera_view_matrix(self.sim, self.envs[0], self.camera_handles[0][0])))
        proj = np.array(self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0][0]))

        fu = 2 / proj[0, 0]
        fv = 2 / proj[1, 1]

        # Get image dimensions
        height, width = seg_image.shape
        center_u = width / 2
        center_v = height / 2

        # Create grid for image-space coordinates
        u = np.arange(width) - center_u
        v = np.arange(height) - center_v
        uu, vv = np.meshgrid(u, v)

        # Normalize image-space coordinates and apply depth scaling
        uu = uu / width
        vv = vv / height
        d = depth_image

        # Compute 3D camera coordinates for each pixel (all at once)
        X2 = np.stack((d * fu * uu, d * fv * vv, d, np.ones_like(d)), axis=-1)
        
        # Transform points to world coordinates
        p2 = X2 @ vinv.T  # Apply inverse camera view to get world coordinates
        points = np.stack((p2[..., 2], p2[..., 0], p2[..., 1]), axis=-1)

        # Filter points based on segmentation image
        mask_tool = seg_image == 1
        mask_ball = seg_image == 2
        mask_bowl = seg_image == 4

        filter_tool = points[mask_tool]
        filter_ball = points[mask_ball]
        filter_bowl = points[mask_bowl]

        # Concatenate and label points
        filter_points = np.concatenate((filter_tool, filter_ball, filter_bowl), axis=0)
        filter_seg = np.concatenate(([0] * len(filter_tool), [1] * len(filter_ball), [2] * len(filter_bowl)))

        # Stack points with labels
        seg_pcd = np.concatenate((filter_points, filter_seg[:, np.newaxis]), axis=1)
        
        # Align the point cloud if needed
        seg_pcd = self.align_point_cloud(seg_pcd)

        return seg_pcd


    def compute_camera_intrinsics_matrix(self, image_width = 320, image_heigth = 240, horizontal_fov = 57):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

        K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], device=self.device, dtype=torch.float)

        return K
    
    def normalize_image(self, img):
   
        min_val = -1.1149625
        max_val = -0.46974927

        # Normalize the image to 0-1 range
        normalized_img = (img - min_val) / (max_val - min_val)
        normalized_img = 1- normalized_img

        
        return normalized_img

    def list_to_nparray(self, lists):
        temp_array = []


        for i in range(len(lists)):
            temp_array.append(np.array(lists[i]))

        temp = np.stack(temp_array, axis=0)

        return temp
    
    def align_point_cloud(self, points, target_points=4000):
        num_points = len(points)

        if num_points >= target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
            indices = np.sort(indices)

        else:
            # Resample with replacement to reach target_points
            indices = np.random.choice(num_points, target_points, replace=True)
            indices = np.sort(indices)

        new_pcd = np.asarray(points)[indices]
        
        return new_pcd
    
    def write_file(self):
        # Specify the file name
        file_name = "spillage.json"

        try:
            # Read existing data from the file
            with open(file_name, "r") as file:
                data = json.load(file)  # Load JSON as a Python dictionary
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize default data if file doesn't exist or is corrupted
            data = {
                "Ball Radius": self.ball_radius,
                "Ball Mass": self.ball_mass,
                "Ball Friction": self.ball_friction,
                "Ball Amount": self.ball_amount,
                "Spillage": 0,
            }

        # Update the dictionary with new values

        data["Spillage"] += self.total_spillage

        # Write the updated dictionary back to the file
        with open(file_name, "w") as file:
            json.dump(data, file, indent=4)  # Use indent for pretty formatting

        print(f"Updated data written to {file_name}")




class ProbabilityVisualizer:
    def __init__(self, prob_array):
    
        # Store the probability array and initialize the plot
        self.prob_array = prob_array
        self.fig, self.ax = plt.subplots()
        self.bars = self.ax.bar(['spillage'], [0])  # Single bar initialized at 0

        # Set the y-axis limit from 0 to 1
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probability")

    def update_bar(self, frame):
        # Get the probability for the current frame
        prob = self.prob_array[frame]
        
        # Update the height of the bar
        self.bars[0].set_height(prob)
        
        # Optionally, update the title to display the current probability
        self.ax.set_title(f"Spillage Probability: {prob:.2f}")

    def animate(self):
        # Create animation over the range of the probability array
        ani = FuncAnimation(self.fig, self.update_bar, frames=len(self.prob_array), interval=500, repeat=False)
        save_path = "test.gif"
        if save_path:
            if save_path.endswith('.gif'):
                writer = PillowWriter(fps=1)  # GIF writer
                ani.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
      
        # plt.show()

if __name__ == "__main__":
    
    prob_list = []
    gd_list = []

    # Initialize IsaacSim
    issac = IsaacSim()
    # Perform the scooping process
    prob_array, gd_spillage = issac.scooping_process()
    # Append results
    prob_list.append(prob_array)
    gd_list.append(gd_spillage)
    
    '''
    # print(np.sum(gd_list, axis=0))
    # print(np.sum(prob_list, axis=0))
    mean_prob = np.mean(np.stack(prob_list), axis=0)
    mean_gd = np.mean(np.stack(gd_list), axis=0)

    # np.save('prob_array.npy', np.array(prob_array))

    # prob_array = np.load('array.npy')
    # visualizer = ProbabilityVisualizer(prob_array)
    # visualizer.animate()


    # Create an x-axis for the number of data points
    x = np.arange(len(mean_prob))
    plt.ylim(0, 1)

    # Plot both arrays on the same chart
    plt.plot(x, mean_gd, marker='o', linestyle='-', color='b', label='gd_spillage')
    plt.plot(x, mean_prob, marker='x', linestyle='--', color='r', label='prob_array')

    # Add labels, title, and legend
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.legend()

    # plt.savefig("probability_chart.png", format="png", dpi=300)  # Set format and resolution (dpi)

    # Show the plot
    plt.show()
    '''