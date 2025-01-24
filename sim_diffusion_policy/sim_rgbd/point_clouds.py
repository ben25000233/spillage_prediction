import torch
import numpy as np
import open3d as o3d


# create o3d point clouds
def depth_image_to_point_cloud(
    rgb, depth, intrinsic_matrix, depth_scale=1, remove_outliers=True, z_threshold=None, mask=None, device="cuda:0"
):
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

    # remove pcd outliers
    if remove_outliers:
        scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.01)

    return np.asarray(scene_pcd.points), np.asarray(scene_pcd.colors), scene_pcd
