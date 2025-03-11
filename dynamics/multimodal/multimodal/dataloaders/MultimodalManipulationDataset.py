import h5py
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d,  matrix_to_quaternion, quaternion_to_matrix



class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset with lazy loading."""

    def __init__(
        self,
        filename_list,
        data_length=50,
        training_type="selfsupervised",
        action_dim=7,
        single_env_steps=None, 
        type="train",
    ):
        """
        Args:
            filename_list (list): List of paths to HDF5 files.
            data_length (int): Length of the data sequence minus one.
            training_type (str): Type of training (e.g., selfsupervised).
            action_dim (int): Dimension of the action space.
            single_env_steps (int): Not used in this implementation.
            type (str): Type of dataset (e.g., 'train' or 'test').
        """

        


        self.single_env_steps = single_env_steps
        self.dataset_path = filename_list
        self.data_length_in_eachfile = data_length 
        self.training_type = training_type
        self.action_dim = action_dim
        self.type = type
        print(len(filename_list))
        print(filename_list[0])
     
        # We no longer load all the data at once
        self.file_handles = [h5py.File(file, 'r') for file in filename_list]
        print(len(self.file_handles))
        exit()

        

    def __len__(self):
        return len(self.dataset_path) * self.data_length_in_eachfile

    def __getitem__(self, idx):

        # Determine which file and which data entry within that file to load
        file_idx = idx // self.data_length_in_eachfile
        data_index = idx % self.data_length_in_eachfile

        # Open the corresponding file and load the specific data entry
        
        dataset = self.file_handles[file_idx]
        data = self._read_data_from_file(dataset, data_index)

        return data

    def _read_data_from_file(self, dataset, idx):
 
        # Read data from a single file for the given index

        # single_num : 8
        # single_num = len(dataset["tool_ball_bowl_pcd"]) / (self.data_length_in_eachfile + 1)
        # print(single_num)
        # exit()
        # current_index = int(idx * single_num) 


        # total predict prame : look back(2) add current frame
        look_back_frame = 5
     
        if idx == 0:
            tool_ball_bowl_pcd = np.tile(dataset["tool_ball_bowl_pcd"][0] , (look_back_frame, 1, 1)).astype(np.float32)
            
        
        else : 
            begin_idx = -look_back_frame 
            tool_ball_bowl_pcd = dataset["tool_ball_bowl_pcd"][(idx-1)*8 : idx*8][begin_idx:].astype(np.float32)

        self.check_pcd_color(tool_ball_bowl_pcd)
        exit()
        # future ee_pose and tool_pcd
        future_eepose_num = 8 
        eepose = dataset["eepose"][idx*8: (idx+1)*8]
        

        # exit()

        binary_label = dataset["binary_spillage"][idx]
        
        rotation_6d_pose = qua_to_rotation_6d(eepose)
        
        #future ee_pcd
        # self.check_pcd_color(tool_ball_bowl_pcd[0])
        # exit()
        # print(tool_ball_bowl_pcd[0].shape)
        # min_index = np.argmin(tool_pcd[0][:, 0])
        # eepoint1 = tool_pcd[0][min_index]

        # trans_pcd = self.cal_transformation(eepose[0], eepose[-1], tool_pcd[0],eepoint1)
        # flow = trans_pcd[:, :3] - tool_pcd[0][:, :3]
      
        # pcd_with_flow = np.concatenate((tool_pcd[0], flow), axis = 1)
        
        # self.show_arrow(tool_pcd[0], tool_pcd[-1],trans_pcd, eepose[0])
        
      
        single_data = {
            "eepose": rotation_6d_pose,
            # "ee_pcd" : tool_pcd,
            # "spillage_type": spillage_index,
            # "tool_with_ball_pcd" : tool_with_ball_pcd, 
            # "pcd_with_flow" : pcd_with_flow,
            "binary_label" : binary_label,
            "tool_ball_bowl_pcd" : tool_ball_bowl_pcd,
        }

        return single_data
    

    def cal_transformation(self, p1, p2, pcd, eepoint1):

        pos1 = p1.copy()
        pos2 = p2.copy()

        bias = [-0.15,-0.15,0.03]
        temp = pos1[:3] - (eepoint1[:3] - bias) 
        pos1[:3] -= temp
        pos2[:3] -= temp
        
        pcd_point = pcd[:, :3]
        
        pcd_seg = pcd[:,3]
        from scipy.spatial.transform import Rotation as R
        r1 = pos1[:4]
        r2 = pos2[:4]
        # r1 = np.array([pose1[3], pose1[0],pose1[1],pose1[2]])
        # r2 = np.array([pose2[3], pose2[0],pose2[1],pose2[2]])
        rot1 = R.from_quat(r1).as_matrix()  # Rotation matrix from quat1
        rot2 = R.from_quat(r2).as_matrix()  # Rotation matrix from quat2
        # Compute the relative transformation
        relative_rotation = rot2 @ rot1.T      # Relative rotation matrix
   
        # relative_rotation[0, 2] = -relative_rotation[0, 2]  # Negate sin(theta)
        # relative_rotation[2, 0] = -relative_rotation[2, 0]  # Negate -sin(theta)

        base_point = pos1[:3]
        pcd_point = pcd_point - base_point
       
        relative_translation = pos2[:3] - (relative_rotation @ pos1[:3])  # Relative translation

        transformed_point = (relative_rotation @ pcd_point.T).T  + relative_translation + base_point
  
        trans_pcd = np.concatenate((transformed_point, pcd_seg.reshape(pcd_seg.shape[0], 1)), axis = 1)

        return trans_pcd

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
        

    def show_arrow(self, pcd1, pcd2, pcd3, eepose):
        
        pcd1_points = pcd1[:,:3]
        pcd2_points = pcd2[:,:3]
        pcd3_points = pcd3[:,:3]

   
        

        check_num = pcd1_points.shape[0]

        point_cloud_1 = o3d.geometry.PointCloud()
        point_cloud_1.points = o3d.utility.Vector3dVector(pcd1_points[:check_num])

        point_cloud_2 = o3d.geometry.PointCloud()
        point_cloud_2.points = o3d.utility.Vector3dVector(pcd2_points[:check_num])

        point_cloud_3 = o3d.geometry.PointCloud()
        point_cloud_3.points = o3d.utility.Vector3dVector(pcd3_points[:check_num])
  
        point_cloud_4 = o3d.geometry.PointCloud()
        eepoint = eepose[:3]

        # min_index = np.argmin(pcd1[:, 0])
        # eepoint = pcd1[min_index]
        

        p1 = p2 = p3 = np.array([eepoint[0], eepoint[1], eepoint[2]])
        p1 = p1 - 0.01
  

   
        points = np.array([p1, p2, p3])
     
        point_cloud_4.points = o3d.utility.Vector3dVector(points)


        point_cloud_1.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (check_num, 1)))  # Red
        point_cloud_2.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (check_num, 1)))  # Blue
        point_cloud_3.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (check_num, 1)))  #Green
        point_cloud_4.colors = o3d.utility.Vector3dVector([[0, 0, 1], [0,0,1], [0,0,1]])  
        # Visualize point clouds
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add point clouds
        vis.add_geometry(point_cloud_1)
        # vis.add_geometry(point_cloud_2)
        vis.add_geometry(point_cloud_3)
        # vis.add_geometry(point_cloud_4)


        
        '''
        
        for i in range(pcd1_points.shape[0]):
            # Calculate the vector difference and its magnitude (length of the arrow)
            vector_diff = pcd3_points[i] - pcd1_points[i]
            arrow_length = np.linalg.norm(vector_diff)  # This is the desired length of the arrow

            if arrow_length <= 1e-6:
                continue

            
            # Create the arrow with a cylinder height proportional to the distance between pcd1[i] and pcd2[i]
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.0002,  # You can adjust the radius as needed
                cone_radius=0.002,      # Adjust cone radius as well
                cylinder_height=arrow_length/2,  # Set the length of the arrow to the vector difference
                cone_height=0.01  # The cone height can remain constant
            )

            # Translate the arrow to the start point pcd1[i]
            arrow.translate(pcd1_points[i])  # Use only the first 3 elements (x, y, z)

            # Normalize the direction vector
            direction = vector_diff / arrow_length  # Normalize to unit vector
            arrow_direction = np.array([0, 0, 1])  # Default arrow direction in Open3D (pointing along Z-axis)

            # Compute the rotation matrix to align the arrow to the vector difference
            cross_prod = np.cross(arrow_direction, direction)
            cross_prod_norm = np.linalg.norm(cross_prod)
            dot_prod = np.dot(arrow_direction, direction)

            if cross_prod_norm < 1e-6:  # If the vectors are aligned, no need to rotate
                rotation_matrix = np.eye(3)
            else:
                skew_symmetric = np.array([[0, -cross_prod[2], cross_prod[1]],
                                        [cross_prod[2], 0, -cross_prod[0]],
                                        [-cross_prod[1], cross_prod[0], 0]])
                rotation_matrix = (np.eye(3) + skew_symmetric +
                                np.matmul(skew_symmetric, skew_symmetric) * 
                                (1 - dot_prod) / (cross_prod_norm ** 2))

            # Apply the rotation to the arrow to align it with the vector difference
            arrow.rotate(rotation_matrix, center=pcd1_points[i])

            # Add the arrow to the visualization
            vis.add_geometry(arrow)

            if i == check_num:
                break
        
        '''
        # Start visualization
        vis.run()
        vis.destroy_window()
        


    def __del__(self):
        # Close all file handles when the dataset object is deleted
        for file_handle in self.file_handles:
            file_handle.close()


def get_translation_matrix(x, y, z):
    return torch.tensor([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
    ])

def qua_to_rotation_6d(ee_traj):

    ee_traj = torch.tensor(ee_traj, dtype=torch.float32)

    rotation_6d_traj = torch.zeros((ee_traj.shape[0], 9))  # (H, 9)
    
    for i in range(ee_traj.shape[0]):
        ee = ee_traj[i]
        quaternion = ee[3:]  # (qw, qx, qy, qz)

        # Convert quaternion to rotation matrix
        ee_rotation_matrix = quaternion_to_matrix(quaternion)

        # adjust_matrix = R.from_quat(quaternion).as_matrix()

        # print(ee_rotation_matrix)
        # print(adjust_matrix)
        # exit()


        rotation_6d_traj[i, :3] = ee[:3]
        rotation_6d_traj[i, 3:] = matrix_to_rotation_6d(ee_rotation_matrix[0:3, 0:3])

    return np.array(rotation_6d_traj)  # (H, 9)

