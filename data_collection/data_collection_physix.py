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



class IsaacSim():
    def __init__(self,down_trial,scoop_trial, trans_trial):
        self.down_trial = down_trial
        self.scoop_trial = scoop_trial
        self.trans_trial = trans_trial
        
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
        # self.cam_pos = gymapi.Vec3(0.5, 0.1, 0.2)
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, cam_target)
        


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        
        self.All_poses = [torch.tensor([], device=self.device) for _ in range(self.num_envs)]
        
        
        self.All_steps = np.zeros(self.num_envs)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)



        self.action_space = {
            
            "left": torch.Tensor([[-0.0020,  0.0060,  0.0034,  0.0046, -0.0040, -0.0030,  0]]),
            "right" : torch.Tensor([[0.0020, -0.0060, -0.0034, -0.0046,  0.0040,  0.0030, 0]]),
            "forward": torch.Tensor([[0.0041, 0.0006, 0.0023, 0.0003, 0.0025,  -0.0038, 0]]),
            "backward": torch.Tensor([[-0.0041, -0.0006, -0.0023, -0.0003, -0.0025,  0.0038, 0]]),
            "up": torch.Tensor([[-0.0027, -0.0048,  0.0032,  0.0027,  0.0052, -0.0055, 0]]),
            "down": torch.Tensor([[0.0027,  0.0048, -0.0032, -0.0027, -0.0052,  0.0055,  0]]),
            "rest": torch.Tensor([[0.0,  0.0, -0.0, -0.0, -0.0,  0.0,  0]]),
       
        }
        
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

 
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1

        sim_params.physx.friction_offset_threshold = 0.01
        sim_params.physx.friction_correlation_distance = 5
        # lead lag
        sim_params.physx.contact_offset = 0.001
        
        sim_params.physx.rest_offset = 0.000001
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
            body_shape_prop[k].thickness = 0.01
            body_shape_prop[k].friction = 0.001
    
      
     
            
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
        self.ball_radius = round(random.uniform(0.003, 0.008), 4)
        self.ball_mass = round(random.uniform(0.0001, 0.0025), 4)
        self.ball_friction = round(random.uniform(0, 0.2),3)

        self.check_big = False

        # big balls
        big = random.randint(0, 10)
        if big == 3:
            self.ball_radius = round(random.uniform(0.01, 0.03), 4)
            self.ball_mass = round(random.uniform(0.02, 0.05), 4)
            self.check_big = True

        offset = random.randint(1,4)
  
        max_num = int(64/pow(2, (self.ball_radius - 0.003)*1000))+offset
        self.ball_amount = random.randint(1, max_num)
        
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
        body_shape_prop[0].thickness = 0.001       # the ratio of the final to initial velocity after the rigid body collides. 
        
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
            ran = 6
    
        while ball_amount > 0:
            y = -0.18
            for j in range(ran):
                x = 0.47
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
        self.camera_props.width = 320
        self.camera_props.height = 240
        self.camera_props.horizontal_fov = 57
 
        
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
        
        self.top_rgb_list = [[] for _ in range(self.num_envs)]
        self.top_depth_list = [[] for _ in range(self.num_envs)]
        self.top_seg_list = [[] for _ in range(self.num_envs)]
        self.tool_ball_bowl_pcd_list = [[] for _ in range(self.num_envs)]
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
            bowl_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(1.8))
            
            # bowl_pose.p = gymapi.Vec3(0.523, -0.166 , 0.0573)   
            bowl_pose.p = gymapi.Vec3(0.51, -0.15 , 0.026)  

            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0, segmentationId = 4)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            body_shape_prop[0].thickness = 0.0007      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
            body_shape_prop[0].friction = 0.001
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
            table_pose.p = gymapi.Vec3(0.5, -0.5 , -0.2)   
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
            side_cam_pose = np.load('./real_cam_pose/cam2base.npy')

            # 1.004056, -0.047730, 0.766758
            pose = side_cam_pose[:3, 3]

            # [ 0.68139714  0.67507474 -0.2153878  -0.1832488 ]
            rotation_matrix = side_cam_pose[:3, :3]
   
            
            # camera_transform.p = self.cam_pos = gymapi.Vec3(pose[0], pose[1]-0.04, pose[2])
            camera_transform.p = self.cam_pos = gymapi.Vec3(pose[0], pose[1], pose[2])
            #assign
            z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(180))
            y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(55.5))
            x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(-4))

            camera_transform.r = z*y*x
        
            # self.cam_target = gymapi.Vec3(0.5, -0.18, 0.3)
            self.gym.set_camera_transform(camera_1, self.env_ptr, camera_transform)


            #self.gym.set_camera_transform(camera_1, self.env_ptr, camera_transform)
            # print(camera_transform.p)
            # print(camera_transform.r)
          
            self.camera_handles[i].append(camera_1)
            '''
            # add camera_2

            camera_2 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            # camera_offset = gymapi.Vec3(0.01, 0.0, 0.16)
            camera_offset = gymapi.Vec3(0.0565, -0.036, 0.056)

            #camera_rotation = gymapi.Quat(0.8, 0, 0.6, 0)
            # -0.1832488 0.68139714  0.67507474 -0.2153878  
            # degree 276.8
            z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(-2))
            y = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(275))
            x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(0))
            camera_rotation = z*y*x
          
            self.gym.attach_camera_to_body(camera_2, self.env_ptr, self.ee_handles[i], gymapi.Transform(camera_offset, camera_rotation),
                                    gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_2)

            
            #add camera_3
            camera_3 = self.gym.create_camera_sensor(self.env_ptr, self.camera_props)
            camera_transform = gymapi.Transform()
    
            cam_pos = gymapi.Vec3(0.5, 0.15, 0.2)
            cam_target = gymapi.Vec3(0.5, -0.18, 0.323)
            self.gym.set_camera_location(camera_3, self.env_ptr, cam_pos, cam_target)
      
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
            elif z > 0.15 or (z > 0 and y > 0.05)  :
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
            # print(f"spillage amount :{int(spillage_amount - self.pre_spillage[env_index])}")
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

            
    def move_generate(self, franka_idx):

        #init noise
        noise = torch.Tensor([[0,0,0,0,0,0,0]])

        # "down" -> scoop -> trans

        down_F_num = 23
    
        noise -= (self.down_trial[-1] - self.down_trial[down_F_num-1])
        
  
        down_L_index = random.sample(range(down_F_num), random.randint(0, 15))
        down_R_index = random.sample(range(down_F_num), random.randint(0, 10))
        second_down = random.randint(3, 7)
        down_D_index = random.sample(range(down_F_num), second_down)
        down_F_index = random.sample(range(down_F_num), random.randint(0, 3))

        down_list = torch.from_numpy(self.down_trial[:down_F_num]).clone()
        all_down = down_F_num + second_down

        for i in down_L_index:
            down_list[i:] = down_list[i:].unsqueeze(0) + self.action_space.get("left")
            noise += self.action_space.get("left")
        for i in down_R_index:
            down_list[i:] = down_list[i:].unsqueeze(0) + self.action_space.get("right")
            noise += self.action_space.get("right")
        for i in down_D_index:
            down_list[i:] = down_list[i:].unsqueeze(0) + self.action_space.get("down")
            noise += self.action_space.get("down")
        for i in down_F_index:
            down_list[i:] = down_list[i:].unsqueeze(0) + self.action_space.get("forward")
            noise += self.action_space.get("forward")
        
        
        # down -> "scoop" -> trans
        scoop_list = noise + self.scoop_trial

        # calibrate
        calibrate_index = random.sample(range(len(scoop_list)), 35 - all_down)
        # scoop ram : L, R, back, up
        scoop_L_index = random.sample(range(len(scoop_list)), random.randint(0, 10))
        scoop_R_index = random.sample(range(len(scoop_list)), random.randint(0, 10))
        scoop_B_index = random.sample(range(len(scoop_list)), random.randint(0, 10))
        scoop_F_index = random.sample(range(len(scoop_list)), random.randint(2, 4))
        scoop_U_index = random.sample(range(len(scoop_list)), random.randint(0, 1))


        for i in calibrate_index:
            scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("left")*2
            scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("forward")*1.5
            noise += self.action_space.get("left")
            noise += self.action_space.get("forward")
        for i in scoop_L_index:
            scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("left")
            noise += self.action_space.get("left")
        for i in scoop_R_index:
            scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("right")
            noise += self.action_space.get("right")
        for i in scoop_B_index:
            scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("backward")
            noise += self.action_space.get("backward")
        for i in scoop_F_index:
            scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("forward")
            noise += self.action_space.get("forward")
        # for i in scoop_U_index:
        #     scoop_list[i:] = scoop_list[i:].unsqueeze(0) + self.action_space.get("up")
        #     noise += self.action_space.get("up")

        all_process = torch.cat((down_list,scoop_list), dim = 0)


        # future 8 eepose
        self.execute_action_horizon = 8
        
        all_process = all_process[1:len(all_process) - len(all_process)%self.execute_action_horizon + 2]

        self.All_poses[franka_idx] = all_process
        self.All_steps[franka_idx] = len(all_process)
      
        
    def reset_franka(self, franka_idx, init = 0):

        franka_init_pose = np.append(self.down_trial[0], 0.02)
        franka_init_pose = np.append(franka_init_pose, 0.02)
        franka_init_pose = torch.tensor(franka_init_pose, dtype=torch.float32)

       
        if init == 1:
            self.dof_state[:, self.franka_dof_index, 0] = franka_init_pose
            self.dof_state[:, self.franka_dof_index, 1] = 0

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(self.franka_actor_indices),
                len(self.franka_actor_indices)
            )
            
        else : 
            
            self.dof_state[franka_idx, self.franka_dof_index, 0] = franka_init_pose
            self.dof_state[franka_idx, self.franka_dof_index, 1] = 0
            

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(self.franka_actor_indices[franka_idx].unsqueeze(0)),
                1
            )

    def data_collection(self):

        self.collect_time = 1
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)
        self.slow_time = 6
        self.frame = 0
       
        for i in range(self.num_envs):
            self.move_generate(i)

        while not self.gym.query_viewer_has_closed(self.viewer):
     
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            if self.frame <= 95 : 
                if self.frame == 95 :
                    self.reset_franka(0 , init = 1)

            else :   

                dpose = torch.stack([pose[self.dpose_index[i]] for i, pose in enumerate(self.All_poses)])

                # for slow action
                if self.frame % self.slow_time == 1:
                    self.dpose_index+=1
                    self.All_steps -= 1
                 
                
                self.pos_action[:, :7] = dpose
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
                if self.frame == 95:
                    self.cal_spillage_scooped(i, reset = 1) 
  
                # store infomation with each index, begin with 96th frame 
                if self.frame % self.slow_time == 0 and self.frame > 95:
                    self.get_info(i)
        
                
                
                # calculate the spillage amount
                if self.dpose_index[i] % self.execute_action_horizon == 0 and self.frame % self.slow_time == 3 and self.dpose_index[i] != 0:
                    self.cal_spillage_scooped(i, reset = 0)

                # complete single trial and reset franka
                if self.All_steps[i] == 0:
                    self.round[i] += 1
                    self.move_generate(i)
                    self.dpose_index[i] = 0
                    self.reset_franka(i, init = 0)
                    self.frame = 0
                

            if all(round >= (self.collect_time) for round in self.round):

                tool_ball_bowl_pcd = self.list_to_nparray(self.tool_ball_bowl_pcd_list)
                eepose = self.list_to_nparray(self.record_ee_pose)
                
                spillage_amount = self.list_to_nparray(self.spillage_amount)
                scoop_amount = self.list_to_nparray(self.scooped_amount)
                spillage_vol = self.list_to_nparray(self.spillage_vol)
                scoop_vol = self.list_to_nparray(self.scooped_vol)
                binary_spillage = self.list_to_nparray(self.binary_spillage)
                binary_scoop = self.list_to_nparray(self.binary_scoop)

                if sum(binary_spillage) > 2 :
                    weight_spillage = np.ones(8)
                else : 
                    weight_spillage = binary_spillage
             
            
                eepose = np.array([eepose[i:i+9] for i in range(0, 80, 8)])
                tool_ball_bowl_pcd = np.array([tool_ball_bowl_pcd[i:i+9] for i in range(0, 80, 8)])
          
                
                data_dict = {
                    'eepose' : eepose,
                    'tool_ball_bowl_pcd' : tool_ball_bowl_pcd,
                    'spillage_amount': spillage_amount,
                    'spillage_vol': spillage_vol,
                    'binary_spillage' : binary_spillage,
                    'scoop_amount': scoop_amount,
                    'scoop_vol': scoop_vol,
                    'binary_scoop' : binary_scoop,
                    'weight_spillage' : weight_spillage,
                }
          
                # store the data
                with h5py.File(f'{f"/media/hcis-s22/data/physix_dataset/dataset/time_{self.count}"}.h5', 'w') as h5file:
                    for key, value in data_dict.items():
                        h5file.create_dataset(key, data=value)
                
                break
                
                
            
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
            ee_pose_arr  = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z])
            self.record_ee_pose[env_index].append(ee_pose_arr)

            #get top_pcd
            top_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_DEPTH)
            top_seg_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_SEGMENTATION)
            nor_top_depth = self.normalize_image(top_depth_image[30:135, 50:165])
            top_pcd = self.depth_to_point_cloud(env_index, nor_top_depth,top_seg_image[30:135, 50:165], type = 0)
            
            self.tool_ball_bowl_pcd_list[env_index].append(top_pcd)
    
    
 
            

    def normalize_image(self, img):
        
        min_val = -1.1149625
        max_val = -0.46974927

        # Normalize the image to 0-1 range
        normalized_img = (img - min_val) / (max_val - min_val)
        normalized_img = 1- normalized_img

        
        return normalized_img

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

        # self.check_pcd_color(seg_pcd)

        
        return seg_pcd
    

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
    
        
    def keyboard_control(self):
        # keyboard event
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "scoop_up")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "scoop_down")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "test")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "save")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "quit")




if __name__ == "__main__":
    
    down_trial = np.load("./sample_trial/down_trial.npy")
    scoop_trial = np.load("./sample_trial/scoop_trial.npy")
    trans_trial = np.load("./sample_trial/trans_trial.npy")
  

    issac = IsaacSim(down_trial, scoop_trial, trans_trial)
    issac.data_collection()
