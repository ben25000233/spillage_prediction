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
from scipy.spatial.transform import Rotation as R

from BallGenerator import BallGenerator
from SoftGenerator import SoftGenerator

from scipy.spatial.transform import Rotation as Rot


class IsaacSim():
    def __init__(self, mean_traj, mean_eepose_euler, mean_eepose_qua):

        self.mean_traj = mean_traj
        self.mean_eepose_euler = mean_eepose_euler
        self.mean_eepose_qua = mean_eepose_qua

        self.grain_type = "solid"
        
        self.default_height = 0.2
        #tool_type : spoon, knife, stir, fork
        self.tool = "spoon"


        self.config_file = "./collect_time.yaml"

        with open(self.config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.count = int(self.config['count'])

        # if not os.path.exists("./dynamics/collected_dataset_with_pcd/dataset"):
        #     os.makedirs("./dynamics/collected_dataset_with_pcd/dataset")

        

        # Update self.file_root with the new count
        # self.file_root = f"./dynamics/collected_dataset_with_pcd/dataset_info/time_{self.count}"
        # if not os.path.exists(f"{self.file_root}"):
        #     os.makedirs(f"{self.file_root}")

        # if not os.path.exists(f"{self.file_root}/top_view"):
        #     os.makedirs(f"{self.file_root}/top_view")

        # if not os.path.exists(f"{self.file_root}/hand_view"):
        #     os.makedirs(f"{self.file_root}/hand_view")

        

        count = int(self.count) + 1
        self.config['count'] = count
        #self.config['count'] = 0

        with open(self.config_file, 'w') as file:
            yaml.safe_dump(self.config, file)
        
        
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

        # Look at the first env
        cam_target = gymapi.Vec3(0.5, -0.18, 0.3)
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

        self.execute_action_horizon = 12

        self.action_space = {
            
            "left": torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]),
            "right" : torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]),
            "forward": torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]),
            "backward": torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]),
            "up": torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]),
            "down": torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]),
            "rest": torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]),
        }

        self.gt_front = np.load("./real_cam_pose/front_cam2base.npy")
        self.gt_back = np.load("./real_cam_pose/back_cam2base.npy")
        self.ref_bowl = np.load("./ref_sim_bowl.npy")
        self.init_spoon_pcd = np.load("./init_spoon_pcd.npy")
        self.offset_list = np.load("pcd_offset_list.npy")


    def control_ik(self, dpose, damping=0.05):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)

        return u
    
    def create_sim(self):
        
        # parse arguments
        args = gymutil.parse_arguments()

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

 
        sim_params.dt = 1.0/60
        sim_params.substeps = 10

 
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 4

        sim_params.physx.friction_offset_threshold = 0.00001  # lead 穿模
        sim_params.physx.friction_correlation_distance = 1
        # lead lag
        sim_params.physx.contact_offset = 0.00001
        
        sim_params.physx.rest_offset = 0.0001
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

        # file_name = 'bowl/transparent_bowl.urdf'
        # asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        # asset_options.vhacd_enabled = True
        # asset_options.fix_base_link = True
        # self.transparent_bowl_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)

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
        asset_options.vhacd_params.resolution = 10000000
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
        self.franka_handle = self.gym.create_actor(self.env_ptr, self.franka_asset, self.franka_start_pose, "franka",0,4, segmentationId = 1)
       
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.franka_handle)
        for k in range(11):
            body_shape_prop[k].thickness = 5
            body_shape_prop[k].friction = 0.0
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
        self.ball_radius = round(random.uniform(0.003, 0.006), 4)
        self.ball_radius = 0.006
        self.ball_mass = round(random.uniform(0.001, 0.025), 4)
        self.ball_friction = round(random.uniform(0, 0.1),2)

        # self.check_big = False

        # big balls
        # big = random.randint(0, 20)
        # if big == 3:
        #     self.ball_radius = round(random.uniform(0.02, 0.03), 2)
        #     self.ball_mass = round(random.uniform(0.04, 0.07), 4)
        #     self.check_big = True

        offset = random.randint(2,5)
  
        max_num = int(64/pow(2, (self.ball_radius - 0.003)*1000))+offset
        self.ball_amount = random.randint(1, max_num)
        # self.ball_amount = max_num
        # with open(f"{self.file_root}/ball_property" , "a") as file:
        #     file.write(f"radius:{self.ball_radius}\n")
        #     file.write(f"mass:{self.ball_mass}\n")
        #     file.write(f"friction:{self.ball_friction}\n")
        #     file.write(f"amount:{self.ball_amount}\n")



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
        body_shape_prop[0].thickness = 2       # the ratio of the final to initial velocity after the rigid body collides. 
        
        self.gym.set_actor_rigid_shape_properties(self.env_ptr, ball_handle, body_shape_prop)
        c = np.array([115, 78, 48]) / 255.0
        color = gymapi.Vec3(c[0], c[1], c[2])
        color = gymapi.Vec3(1, 1, 1)
        self.gym.set_rigid_body_color(self.env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return ball_handle
    
    def add_solid(self):
        #add balls
        ball_amount = self.ball_amount
        ball_pose = gymapi.Transform()
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.ball_handle = []

        z = 0.1 + self.ball_radius
        if self.check_big == True :
            ran = 1
        else : 
            ran = 5
    
        while ball_amount > 0:
            y = -0.15
            for j in range(ran):
                x = self.bowl_x - 0.03
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
        
        # action index
        self.dpose_index =  np.zeros(self.num_envs ,dtype=int)

        # cache some common handles for later use
        self.camera_handles = [[]for _ in range(self.num_envs)]
        self.franka_indices = []
        self.envs = []
        self.ee_handles = []

        #set camera
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 1280
        self.camera_props.height = 960
        self.camera_props.horizontal_fov = 65
 
        
        #store information
        self.pre_spillage = np.zeros(self.num_envs)

        self.ball_handles = [[] for _ in range(self.num_envs)]
        self.spillage_amount = [[] for _ in range(self.num_envs)]
        self.scooped_amount = [[] for _ in range(self.num_envs)]
        self.spillage_vol = [[] for _ in range(self.num_envs)]
        self.scooped_vol = [[] for _ in range(self.num_envs)]
        self.spillage_type = [[] for _ in range(self.num_envs)]
        self.scooped_type = [[] for _ in range(self.num_envs)]
        self.binary_spillage = [[] for _ in range(self.num_envs)]
        self.binary_scoop = [[] for _ in range(self.num_envs)]

        self.record_ee_pose = [[] for _ in range(self.num_envs)]
        self.isaac_ee_pose = [[] for _ in range(self.num_envs)]

        
        self.top_rgb_list = [[] for _ in range(self.num_envs)]
        self.top_depth_list = [[] for _ in range(self.num_envs)]
        self.top_seg_list = [[] for _ in range(self.num_envs)]
        self.ball_pcd_list = [[] for _ in range(self.num_envs)]
        self.all_pcd_list = [[] for _ in range(self.num_envs)]
        self.front_seg_list = [[] for _ in range(self.num_envs)]

        self.top_pcd_color_list = [[] for _ in range(self.num_envs)]


        self.hand_rgb_list = [[] for _ in range(self.num_envs)]
        self.hand_depth_list = [[] for _ in range(self.num_envs)]
        self.hand_seg_list = [[] for _ in range(self.num_envs)]
        self.hand_pcd_point_list = [[] for _ in range(self.num_envs)]
        self.hand_pcd_color_list = [[] for _ in range(self.num_envs)]


        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)

            
            # add bowl_1
            bowl_pose = gymapi.Transform()
            # bowl_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(1.8))
            
            # bowl_pose.p = gymapi.Vec3(0.523, -0.166 , 0.0573)
            # ori : 0.615
            # 0.45 - 0.65
            # self.bowl_x =  round(random.uniform(0.45, 0.65), 2)
            self.bowl_x = 0.59
            bowl_pose.p = gymapi.Vec3(self.bowl_x, -0.11 , 0.03)

            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0, segmentationId = 4)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            body_shape_prop[0].thickness = 0.01      
            body_shape_prop[0].friction = 0.001
            body_shape_prop[0].rest_offset = 0.001
            self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl_1, body_shape_prop)
            
            
            # #add bowl_2
            # bowl_pose = gymapi.Transform()
            # bowl_pose.r = gymapi.Quat(0, 0, 0, 1)
            # bowl_pose.p = gymapi.Vec3(0.5, 0.1 , 0.1)   
            # self.bowl_2 = self.gym.create_actor(self.env_ptr, self.transparent_bowl_asset, bowl_pose, "bowl_2", 0, 0)

            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl)
            # body_shape_prop[0].thickness = 0.005
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.bowl, body_shape_prop)
            
            
            # add tabel
            table_pose = gymapi.Transform()
            table_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.23)   
            self.table = self.gym.create_actor(self.env_ptr, self.table_asset, table_pose, "table", 0, 0, segmentationId = 3)
            color = gymapi.Vec3(144/255, 164/255, 203/255)
            self.gym.set_rigid_body_color(self.env_ptr, self.table, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            
            # body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.table)
            # body_shape_prop[0].thickness = 0.0005      
            # body_shape_prop[0].friction = 0.5
            # self.gym.set_actor_rigid_shape_properties(self.env_ptr, self.table, body_shape_prop)
            
            #add ball
            self.add_solid()
            self.ball_handles[i] = self.ball_handle

            #add franka
            self.add_franka()

            #add camera_1
            camera_1 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_transform = gymapi.Transform()
            front_cam_pose = np.load('./real_cam_pose/front_cam2base.npy')

            pose = front_cam_pose[:3, 3]

            adjust_matrix = np.eye(4)
            adjust_matrix[0:3, 0:3] = Rot.from_euler("XYZ", (90, 0, 90), degrees=True).as_matrix()

            camera_pose = front_cam_pose @ adjust_matrix
            pose = camera_pose[0:3, 3]
            
            camera_transform.p = self.cam_pos = gymapi.Vec3(*pose)
            camera_transform.r = gymapi.Quat(*Rot.from_matrix(camera_pose[0:3, 0:3]).as_quat())

            self.gym.set_camera_transform(camera_1, self.env_ptr, camera_transform)
            self.camera_handles[i].append(camera_1)

            #add camera_2
            camera_2 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_transform = gymapi.Transform()
            back_cam_pose = np.load('./real_cam_pose/back_cam2base.npy')

            pose = back_cam_pose[:3, 3]

            adjust_matrix = np.eye(4)
            adjust_matrix[0:3, 0:3] = Rot.from_euler("XYZ", (90, 0, 90), degrees=True).as_matrix()

            camera_pose = back_cam_pose @ adjust_matrix
            pose = camera_pose[0:3, 3]
            
            camera_transform.p = gymapi.Vec3(*pose)
            camera_transform.r = gymapi.Quat(*Rot.from_matrix(camera_pose[0:3, 0:3]).as_quat())

            self.gym.set_camera_transform(camera_2, self.env_ptr, camera_transform)
            self.camera_handles[i].append(camera_2)
            '''
            # add camera_3

            camera_3 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_transform = gymapi.Transform()
            pose = front_cam_pose[:3, 3]
            camera_transform.p = gymapi.Vec3(*pose)

            z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(180))
            y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(60))
            x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(0))
          
            camera_transform.r = z*y*x
        

            self.gym.set_camera_transform(camera_3, self.env_ptr, camera_transform)
  
            self.camera_handles[i].append(camera_3)
            '''


        self.franka_actor_indices = to_torch(self.franka_indices, dtype = torch.int32, device = "cpu")


    def cal_spillage_scooped(self, env_index, reset = 0):
        # reset = 1 means record init spillage in experiment setting 
        current_spillage = 0
        scoop_amount = 0

        # calculate spillage and scoop amount
        for ball in self.ball_handles[env_index]:
            body_states = self.gym.get_actor_rigid_body_states(self.envs[env_index], ball, gymapi.STATE_ALL)
            z = body_states['pose']['p'][0][2]
            y = body_states['pose']['p'][0][1]

            if z < 0:
                current_spillage += 1
            elif z > 0.15 or (z > 0 and y > 0)  :
                scoop_amount += 1
        if reset == 0:
         
            spillage_amount = current_spillage - self.pre_spillage[env_index]
            spillage_vol = spillage_amount * (self.ball_radius**3) * 10**9
            scoop_vol = scoop_amount * (self.ball_radius**3)* 10**9
            
            if int(spillage_amount) == 0:
                self.binary_spillage[env_index].append(0)
            else :
                self.binary_spillage[env_index].append(1)
    
            if int(scoop_amount) == 0:
                self.binary_scoop[env_index].append(0)
            else :
                self.binary_scoop[env_index].append(1)
          

            self.spillage_amount[env_index].append(int(spillage_amount))
            self.scooped_amount[env_index].append(int(scoop_amount))
            self.spillage_vol[env_index].append(int(spillage_vol))
            self.scooped_vol[env_index].append(int(scoop_vol))
            '''
            if (self.dpose_index[env_index]+1) % 80 == 0 : 
                split_spillage = sum(self.spillage_vol[env_index][-10:])
                split_scoop = self.scooped_vol[env_index][-1]
                spillage_index = self.define_scale(split_spillage, type = "spillage")
                scoop_index = self.define_scale(split_scoop, type = "scoop")

                for i in range(0, 10):
                    self.spillage_type[env_index].append(spillage_index) 
                    self.scooped_type[env_index].append(scoop_index)

                # print(f"spillage_amount : {split_spillage}")
                # print(f"spillage_type : {self.spillage_type[env_index][-1]}")
                # print(f"scoop : {self.scooped_type[env_index][-6:]}")
            '''

            # print(self.dpose_index[env_index])
            # print(f"spillage amount :{int(spillage_amount)}")
            # print(f"spillage vol : {int(spillage_vol)}")
            # print(f"scoop_num : {int(scoop_amount)}")
            # print(f"scoop_vol : {int(scoop_vol)}")
        self.pre_spillage[env_index] = int(current_spillage)


    def define_scale(self, amount, type = None):
        
        spillage_range = [0, 3000]

        # < 500 : strongly scoop more
        # 500-2000 : weakly scoop more
        # 2000 - 4000 : well done
        # 4000 - 6000 : weakly scoop less
        # > 6000 : strongly scoop less
        scoop_range = [0, 5, 4000, 6000]

        if amount == 0:
            index = 0
        else :
            if type == "scoop":
                index = sum([amount > r for r in scoop_range[1:]]) + 1
                
            elif type == "spillage" :
                index = sum([amount > r for r in spillage_range[1:]]) + 1

        return index


    def offset_generate(self):
        

        self.U_index = np.array(random.sample(range(len(self.mean_traj)), random.randint(0, 30))) 
        self.D_index = np.array(random.sample(range(60), random.randint(0, 30)))+15
        self.L_index = np.array(random.sample(range(len(self.mean_traj)), random.randint(0, 30)))
        self.R_index = np.array(random.sample(range(len(self.mean_traj)), random.randint(0, 30))) 
        self.B_index = np.array(random.sample(range(len(self.mean_traj)), random.randint(0, 30))) 
        # self.F_index = np.array(random.sample(range(len(self.mean_traj)), random.randint(0, 2))) 
        


    def apply_offset(self, index):

        offset = self.control_ik(torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]))
 
        if index in self.L_index :
            offset += self.control_ik(self.action_space.get("left"))/10
        
        if index in self.R_index :
            offset += self.control_ik(self.action_space.get("right"))/10

        if index in self.D_index :
            offset += self.control_ik(self.action_space.get("down"))/10

        # if index in self.F_index :
        #     offset += self.control_ik(self.action_space.get("forward"))

        if index in self.B_index :
            offset += self.control_ik(self.action_space.get("backward"))/10

        # if index in self.U_index :
        #     offset += self.control_ik(self.action_space.get("up"))

        
   
        return np.array(offset.squeeze(0).squeeze(-1))


        
    def reset_franka(self, reset_move = 0):

        if reset_move == 1:
            offset = np.array(self.control_ik(self.action_space.get("left"))*0.001)
            self.mean_traj -= offset
        if reset_move == 2:
            offset = np.array(self.control_ik(self.action_space.get("right"))*0.001)
            self.mean_traj -= offset
        else :
            self.origin_pose = self.mean_eepose_euler[0]

        franka_init_pose = self.mean_traj[0]

        franka_init_pose = np.append(franka_init_pose, 0.02)
        franka_init_pose = np.append(franka_init_pose, 0.02)
        franka_init_pose = torch.tensor(franka_init_pose, dtype=torch.float32)

        self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
        self.dof_state[:, self.franka_dof_index, 1] = 0

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.franka_actor_indices),
            len(self.franka_actor_indices)
            )
            
        

    def data_collection(self):

        self.collect_time = 1
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)
        self.slow_time = 6
        self.frame = 0
        self.start_record = 0
        x_difference = 1
        init_check = 0
        init_spillage_check = 0
        ori_x_difference = 0
        
        go_to_final = False
        check_final_times = 180
        
        for i in range(self.num_envs):
            # self.move_generate(i)
            self.offset_generate()

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            

            if self.frame <= 95 : 
                if self.frame == 95 :
                    self.reset_franka(reset_move = 0)
                    # print()
            
            
            elif go_to_final :
                offset = np.array(self.control_ik(self.action_space.get("forward"))*0.01)

                if self.frame % 12 == 1:
                    franka_final_pose += offset

                self.pos_action[:, :7] = torch.tensor(franka_final_pose)
                self.pos_action[:, 7:] = 0.02

                self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()


                check_final_times -= 1
                if check_final_times == 0:
                    #get top_pcd
                    top_rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_COLOR)
                    top_depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_DEPTH)
                    top_seg_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_SEGMENTATION)

                    last_ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
                    last_isaac_ee_pose = np.array([last_ee_pose.p.x, last_ee_pose.p.y, last_ee_pose.p.z, last_ee_pose.r.w, last_ee_pose.r.x, last_ee_pose.r.y, last_ee_pose.r.z])
                    
                    isaac_quat_rotation_xyzw = np.array([last_ee_pose.r.x, last_ee_pose.r.y, last_ee_pose.r.z, last_ee_pose.r.w])
                    rotation = Rot.from_quat(isaac_quat_rotation_xyzw)
                    rotation_matrix = rotation.as_matrix()
                    z_rot = rotation_matrix[:, 2]

                    updata_eepose = [last_ee_pose.p.x, last_ee_pose.p.y, last_ee_pose.p.z] + z_rot*0.1
                    trans_last_eepose  = np.array([updata_eepose[0], updata_eepose[1], updata_eepose[2], last_ee_pose.r.w, last_ee_pose.r.x, last_ee_pose.r.y, last_ee_pose.r.z])




                    a, b = top_rgb_image.shape
                    crop_rgb = top_rgb_image.reshape(a, int(b/4), 4)[:, :, :3]

                    c, d = top_depth_image.shape
                    crop_depth = top_depth_image.reshape(c, d, 1)*-1000

                    crop_seg = top_seg_image

                    eepose = self.list_to_nparray(self.record_ee_pose)

                    front_seg = self.list_to_nparray(self.front_seg_list)

                    isaac_eepose = self.list_to_nparray(self.isaac_ee_pose)
                  
                
                    data_dict = {
                    'rgb' : crop_rgb,
                    'depth' : crop_depth,
                    'scoop_amount': scoop_amount[-1],
                    'scoop_vol': scoop_vol[-1],
                    'trans_ee_pose_qua_list': eepose,
                    'radius' : self.ball_radius,
                    'mass' : self.ball_mass,
                    'friction' : self.ball_friction,
                    'amount' : self.ball_amount,
                    'all_front_seg_list' : front_seg,
                    'isaac_ee_pose_list' :isaac_eepose,
                    'last_eepose_isaac' : last_isaac_ee_pose,
                    'last_eepose_trans' : trans_last_eepose,
                    'last_seg' : crop_seg,
                    }
                    if scoop_amount[-1] > 0:
                        with h5py.File(f'{f"/media/hcis-s25/data/GRITS/sim_spillage_dataset/last_frame/time_{self.count}"}.h5', 'w') as h5file:
                            for key, value in data_dict.items():
                                h5file.create_dataset(key, data=value)
                    
                    break

            else :   
                    
                if init_check == 0:
                    self.start_record = 1
                    self.mean_eepose_qua[:, 0] -= ori_x_difference
                    init_check = 1
                    
                # for slow action
                if self.frame % self.slow_time == 1:
                    self.dpose_index+=1

                    offset = self.apply_offset(self.dpose_index[0])
                    self.mean_traj[self.dpose_index[0]:] += offset*0.01

                    
                
                self.pos_action[:, :7] = torch.tensor(self.mean_traj[self.dpose_index])
                self.pos_action[:, 7:] = 0.02

                self.dof_state[:, self.franka_dof_index, 0] = self.pos_action.clone()
            
            
            temp = self.dof_state[:, self.franka_dof_index, 0].clone()

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(temp),
                gymtorch.unwrap_tensor(self.franka_actor_indices),
                len(self.franka_actor_indices)
            )

            for i in range(self.num_envs):
            
                # calculate the init spillage amount 
                if self.start_record == 1 and init_spillage_check == 0:
                    self.cal_spillage_scooped(i, reset = 1) 
                    init_spillage_check = 1

                
                # store infomation with each index
                if self.frame % self.slow_time == 0 and self.start_record == 1:
                    self.get_info(i)
                
                            
                
                # calculate the spillage amount
                if self.dpose_index[i] % self.execute_action_horizon == 0 and self.frame % self.slow_time == 3 and self.dpose_index[i] != 0:
                    self.cal_spillage_scooped(i, reset = 0)


                # complete single trial and reset franka
                if self.dpose_index == len(self.mean_traj)-1:
                    
                    self.round[i] += 1
                    self.dpose_index[i] = 0
                    # self.reset_franka(i, init = 0)
                    # self.frame = 0
                
                
            
            if all(round >= (self.collect_time) for round in self.round) and go_to_final == False:
                ball_pcd = self.list_to_nparray(self.ball_pcd_list)
                all_pcd = self.list_to_nparray(self.all_pcd_list)
                eepose = self.list_to_nparray(self.record_ee_pose)
                
                
                spillage_amount = self.list_to_nparray(self.spillage_amount)
                scoop_amount = self.list_to_nparray(self.scooped_amount)
                spillage_vol = self.list_to_nparray(self.spillage_vol)
                scoop_vol = self.list_to_nparray(self.scooped_vol)
                
                binary_spillage = self.list_to_nparray(self.binary_spillage)
                binary_scoop = self.list_to_nparray(self.binary_scoop)

                # if sum(binary_spillage) > 2 :
                #     weight_spillage = np.ones(8)
                # else : 
                #     weight_spillage = binary_spillage
       
         
                
                # [1:] means delete the first init scene
                data_dict = {
                    'eepose' : eepose[1:],
                    'ball_pcd' : ball_pcd[1:],
                    'tool_ball_bowl_pcd' : all_pcd[1:],
                    'spillage_amount': spillage_amount,
                    'spillage_vol': spillage_vol,
                    'binary_spillage' : binary_spillage,
                    'scoop_amount': scoop_amount,
                    'scoop_vol': scoop_vol,
                    'binary_scoop' : binary_scoop,
                    # 'weight_spillage' : weight_spillage,
                    'radius' : self.ball_radius,
                    'mass' : self.ball_mass,
                    'friction' : self.ball_friction,
                    'amount' : self.ball_amount,
                    
                }

                # store the data
                with h5py.File(f'{f"/media/hcis-s25/data/GRITS/sim_spillage_dataset/all_process/time_{self.count}"}.h5', 'w') as h5file:
                    for key, value in data_dict.items():
                        h5file.create_dataset(key, data=value)
                
                
                go_to_final = True
                franka_final_pose = self.mean_traj[-1].reshape(1,7)
            
        
            
            
            # update the viewer
            self.frame += 1
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


    def list_to_nparray(self, lists, type = None):
        temp_array = []

        for i in range(len(lists)):
            temp_array.append(np.array(lists[i]))

        temp = np.stack(temp_array, axis=0)

        shape = temp.shape
        new_shape = (shape[0] * shape[1],) + shape[2:]
        temp_1 = temp.reshape(new_shape )
 
        return temp_1
            

    def get_info(self, env_index):

        # get camera images
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)


        if self.round[env_index] < self.collect_time :

            # get eepose
            ee_pose = self.gym.get_rigid_transform(self.envs[env_index], self.ee_handles[env_index])
                        # sim to real : modify z axis 0.1, y = 0.0105
            isaac_eepose = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])
            isaac_quat_rotation_xyzw = np.array([ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])
            rotation = Rot.from_quat(isaac_quat_rotation_xyzw)
            rotation_matrix = rotation.as_matrix()
            z_rot = rotation_matrix[:, 2]

            updata_eepose = [ee_pose.p.x, ee_pose.p.y, ee_pose.p.z] + z_rot*0.1

            ee_pose_arr  = np.array([updata_eepose[0], updata_eepose[1], updata_eepose[2], ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z])
            self.record_ee_pose[env_index].append(ee_pose_arr)
            self.isaac_ee_pose[env_index].append(isaac_eepose)

            #get top_pcd
            # top_rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_COLOR)
            # top_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_DEPTH)*-1000
            top_seg_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_SEGMENTATION)
            # plt.imshow(top_depth_image[:300, 200:400])
            # plt.show()
            # exit()

            #get back_pcd
 
            back_rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1],  gymapi.IMAGE_COLOR)
            back_depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1],  gymapi.IMAGE_DEPTH)*-1000
            back_seg_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1],  gymapi.IMAGE_SEGMENTATION)

            # get food pcd
            back_food_pcd = self.depth_to_point_cloud(back_depth_image, back_seg_image, object_id = 2)
            back_food_world = self.transform_to_world(back_food_pcd[:, :3], self.gt_back)
            object_seg = np.full((back_food_world.shape[0], 1), 2)
            back_food_world = np.hstack((back_food_world, object_seg))
            back_food_world[:, 2] -= 0.015

            # get tool 
            sim_cor = self.record_ee_pose[0][-1]
            trans_tool = self.from_ee_to_spoon(sim_cor, self.init_spoon_pcd)
            object_seg = np.full((trans_tool.shape[0], 1), 1)
            trans_tool = np.hstack((trans_tool, object_seg))
            
            all_pcd = np.concatenate(( trans_tool, back_food_world, self.ref_bowl), axis=0)

            # self.check_pcd_color(all_pcd)
            # exit()
            
            
            nor_pcd = self.nor_pcd(all_pcd)
            real_pcd = np.load("real_nor_pcd.npy")


            sim_bowl = nor_pcd[nor_pcd[:, 2] == 1]
            nor_pcd[:, 3] = 1
            real_pcd[:, 3] = 2

            all_pcd = np.concatenate(( nor_pcd, real_pcd), axis=0)
            self.check_pcd_color(all_pcd)
            exit()

        
            
            self.all_pcd_list[env_index].append(nor_pcd)
            self.front_seg_list[env_index].append(top_seg_image)

    def compute_camera_intrinsics_matrix(self,image_width, image_heigth, horizontal_fov):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180
        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
        K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]])
        return K

    def transform_to_world(self,points, extrinsic):

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world_homogeneous = (extrinsic @ points_homogeneous.T).T
        points_world = points_world_homogeneous[:, :3]

        return points_world

    def nor_pcd(self, points):

        seg_info = points[:, 3].reshape(len(points), 1)
        points = points[:, :3]

        # normalize the pcd
        
        centroid = np.mean(points, axis=0)
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))


        centroid = [ 0.59115381, -0.1113387 ,  0.0755547 ]
        m = 0.6797932094342392
        
        points = points - centroid
        points = points / m

        seg_pcd = np.concatenate((points, seg_info), axis=1)
        return seg_pcd

    def depth_to_point_cloud(
        self, depth, mask=None,object_id = None, device="cpu"):
        
        mask = (mask == object_id)
    
        intrinsic = self.compute_camera_intrinsics_matrix(1280, 960, 65)
        h, w = depth.shape
    
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        if mask is not None:
            points = points[mask.reshape(-1), :]

        points = self.align_point_cloud(points)

        object_seg = np.full((points.shape[0], 1), object_id)

        # Concatenate the column to the original array
        seg_pcd = np.hstack((points, object_seg))

        # self.check_pcd_color(seg_pcd)

        return seg_pcd
    
    def align_point_cloud(self, points, target_points=10000):
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

    
    def check_pcd_color(self, pcd):

        color_map = {
            0: [1, 0, 0],    # Red
            1: [0, 1, 0],    # Green
            2: [0, 0, 1],    # Blue
            3: [1, 1, 0],    # Yellow
            4: [1, 0, 1]     # Magenta
        }
        points = []
        colors = []
    
        
        for i in range(pcd.shape[0]):
            points.append(pcd[i][:3])
            if pcd.shape[1] == 4:
                colors.append(color_map[pcd[i][3]])


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])
    
    def get_translation_matrix(self, x, y, z):
        return torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
        ])

    def from_ee_to_spoon(self, ee_point, pcd):
        # Convert inputs to torch tensors
        ee_point = torch.tensor(ee_point, dtype=torch.float32)
        pcd = torch.tensor(pcd, dtype=torch.float32)
        offsets = torch.tensor(self.offset_list, dtype=torch.float32)  # Shape: [N, 3]

        # Precompute constants
        adjust_matrix = torch.tensor(
            Rot.from_euler("XYZ", (0, 180, 90), degrees=True).as_matrix(),
            dtype=torch.float32,
        )

        # Create the end-effector pose matrix (4x4)
        ee_pose = torch.eye(4, dtype=torch.float32)
        ee_pose[0:3, 3] = ee_point[0:3]  # Translation
        rotation = Rot.from_quat([ee_point[4], ee_point[5], ee_point[6], ee_point[3]])
        rotation_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32)
        ee_pose[0:3, 0:3] = torch.mm(rotation_matrix, adjust_matrix)

        # Create transformation matrices for offsets (Batch)
        T_spoon_to_center = torch.eye(4, dtype=torch.float32).repeat(len(offsets), 1, 1)  # [N, 4, 4]
        T_spoon_to_center[:, 0:3, 3] = offsets

        # Batch multiply: [N, 4, 4] @ [4, 4] -> [N, 4, 4]
        spoon_poses = torch.matmul(ee_pose.unsqueeze(0), T_spoon_to_center)

        # Extract positions: [N, 3]
        new_pcd = spoon_poses[:, 0:3, 3]

        return new_pcd.numpy()


if __name__ == "__main__":
    

    # [60:150]
    mean_traj = np.load("./sample_trial/mean_traj.npy")[0:150]
    mean_eepsoe_euler = np.load("./sample_trial/mean_eepose_euler.npy")[0:150]
    mean_eepsoe_qua = np.load("./sample_trial/mean_eepose_qua.npy")[0:150]


    issac = IsaacSim(mean_traj, mean_eepsoe_euler, mean_eepsoe_qua)
    issac.data_collection()
