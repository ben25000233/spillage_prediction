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

import transforms3d


class IsaacSim():
    def __init__(self):


        self.grain_type = "solid"
        
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

        self.execute_action_horizon = 8

        self.action_space = {
            
            "left": torch.Tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]),
            "right" : torch.Tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]),
            "forward": torch.Tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]),
            "backward": torch.Tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]),
            "up": torch.Tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]),
            "down": torch.Tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]),
            "rest": torch.Tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]]),
        }

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

 
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 4

        sim_params.physx.friction_offset_threshold = 0.001  # lead 穿模
        sim_params.physx.friction_correlation_distance = 1
        # lead lag
        sim_params.physx.contact_offset = 0.0001
        
        sim_params.physx.rest_offset = 0.001
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
            body_shape_prop[k].thickness = 0.1
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

        
    def create_ball(self):
        self.ball_radius = 0.009
        self.ball_mass = 0.01
        self.ball_friction = 0.0

        self.check_big = False

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
        body_shape_prop[0].thickness = 1       # the ratio of the final to initial velocity after the rigid body collides. 
        
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

        if self.ball_radius > 0.05:
            self.check_big = True

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
        
        self.front_rgb_list = [[] for _ in range(self.num_envs)]
        self.front_depth_list = [[] for _ in range(self.num_envs)]

        self.back_rgb_list = [[] for _ in range(self.num_envs)]
        self.back_depth_list = [[] for _ in range(self.num_envs)]
        


        # create and populate the environments
        for i in range(num_envs):
            # create env
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(self.env_ptr)

            
            # add bowl_1
            bowl_pose = gymapi.Transform()
            self.bowl_x =  0.61
            bowl_pose.p = gymapi.Vec3(self.bowl_x, -0.105 , 0.025)

            self.bowl_1 = self.gym.create_actor(self.env_ptr, self.bowl_asset, bowl_pose, "bowl_1", 0, 0, segmentationId = 4)

            body_shape_prop = self.gym.get_actor_rigid_shape_properties(self.env_ptr, self.bowl_1)
            body_shape_prop[0].thickness = 0.01      # the ratio of the final to initial velocity after the rigid body collides.(but have no idea why it will affect the contact distance) 
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

            from scipy.spatial.transform import Rotation as Rot

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


        self.franka_actor_indices = to_torch(self.franka_indices, dtype = torch.int32, device = "cpu")


    

        
    def reset_franka(self, reset_move = 0):


        franka_init_pose = self.traj[0]

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
            
        

    def data_collection(self, traj):

        self.collect_time = 1
        self.round = [0]*self.num_envs
        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32)
        self.slow_time = 3
        self.frame = 0

        self.traj = traj


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
            
            
            else :   
                    
                # for slow action
                if self.frame % self.slow_time == 1:
                    self.dpose_index+=1

                if self.dpose_index[0] < len(self.traj):
                    self.pos_action[:, :7] = torch.tensor(self.traj[self.dpose_index])
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
                
                # store infomation with each index, begin with 96th frame 
                if self.frame % self.slow_time == 0 and self.frame > 95:
                    self.get_info(i)
                
                
            
            if self.dpose_index[0] == len(self.traj):
                temp = self.list_to_nparray(self.front_depth_list)
                a, b, c = temp.shape

                front_rgb_list = self.list_to_nparray(self.front_rgb_list).reshape(a, b, c, 4)[:, :, :, :3]
                front_depth_list = self.list_to_nparray(self.front_depth_list).reshape(a, b, c, 1)

                back_rgb_list = self.list_to_nparray(self.back_rgb_list).reshape(a, b, c, 4)[:, :, :, :3]
                back_depth_list = self.list_to_nparray(self.back_depth_list).reshape(a, b, c, 1)

                eepose_list = self.list_to_nparray(self.record_ee_pose)


                return front_rgb_list, front_depth_list, back_rgb_list, back_depth_list, eepose_list
                
            
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


            #get top_pcd
            top_rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][0],  gymapi.IMAGE_COLOR)
            top_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][0],  gymapi.IMAGE_DEPTH)

            #get back_pcd
 
            back_rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0][1],  gymapi.IMAGE_COLOR)
            back_depth_image = self.gym.get_camera_image(self.sim, self.envs[env_index], self.camera_handles[env_index][1],  gymapi.IMAGE_DEPTH)
     
            ee_pose = self.gym.get_rigid_transform(self.envs[0], self.ee_handles[0])
            ee_pose_arr  = np.array([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z - 0.1, ee_pose.r.w, ee_pose.r.x, ee_pose.r.y, ee_pose.r.z])
            
            self.record_ee_pose[env_index].append(ee_pose_arr)
            
            self.front_rgb_list[env_index].append(top_rgb_image)
            self.front_depth_list[env_index].append(top_depth_image)

            self.back_rgb_list[env_index].append(back_rgb_image)
            self.back_depth_list[env_index].append(back_depth_image)


    
    

   

if __name__ == "__main__":
    #
    input_type = "leh_05"
    input_path = f"/media/hcis-s22/data/dataset_1_18/{input_type}/joint_states.npy"
    traj = np.load(input_path)[10:]
    print(len(traj))
    exit()
    output_path = f"/media/hcis-s22/data/dataset_1_18/{input_type}"

    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    issac = IsaacSim()
    front_rgb_list, front_depth_list, back_rgb_list, back_depth_list, eepose_list = issac.data_collection(traj)

    np.save(output_path + '/front_rgb.npy', np.array(front_rgb_list))
    np.save(output_path + '/front_depth.npy', np.array(front_depth_list))
    np.save(output_path + '/back_rgb.npy', np.array(back_rgb_list))
    np.save(output_path + '/back_depth.npy', np.array(back_depth_list))
    np.save(output_path + '/ee_pose_qua.npy', np.array(eepose_list))
