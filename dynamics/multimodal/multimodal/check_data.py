import h5py
import os
import numpy as np
from tqdm import tqdm

import open3d as o3d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Path to the folder containing the .h5 files
folder_path = "/media/hcis-s22/data/physix_dataset"
# folder_path = "/home/hcis-s22/benyang/scoop-env/dynamics/new_dataset"

# Get a list of all .h5 files in the folder
h5_files = [f for f in os.listdir(f"{folder_path}/dataset") if f.endswith('.h5')]

tool_centroids_list = []
tool_distances_list = []
tool_with_ball_centroids_list = []
tool_with_ball_distances_list = []

def check_pcd_color(pcd):
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
        colors.append(color_map[pcd[i][3]])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])


distribution = []

for i, file_name in enumerate(tqdm(h5_files)):
    file_path = f"{folder_path}/dataset/{file_name}"
    with h5py.File(file_path, 'r') as dataset:  # 'a' mode opens for read/write without truncating
        mono_dis = []
        mono_dis.append(dataset["radius"][()])
        mono_dis.append(dataset["mass"][()])
        mono_dis.append(dataset["friction"][()])
        mono_dis.append(dataset["amount"][()])
        
    distribution.append(mono_dis)

np_distribution = np.array(distribution)

'''
#plot 2d
plt.scatter(np_distribution[:, 0], np_distribution[:, 1], alpha=0.7, c='blue')
plt.title("Scatter Plot of 2D Vectors")
plt.xlabel("radius")
plt.ylabel("amount")
plt.show()
'''

# PCA to 2D
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(np_distribution)

# Scatter plot
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7, c='blue')
plt.title("PCA Projection of Vectors")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

