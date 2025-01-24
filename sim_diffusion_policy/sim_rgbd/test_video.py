import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

def show_mask(mask, ax, obj_id=None, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.7])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.7])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def check_seg_color(seg_map):
    colormap = np.array([[0, 0, 0],     # Black for 0
                        [255, 0, 0],   # Red for 1
                        [0, 255, 0],   # Green for 2
                         [0, 0, 255]   ])  


    np_seg = seg_map.int().cpu().numpy()
    colored_image = colormap[np_seg]

    plt.imshow(colored_image.astype(np.uint8))
    plt.axis('off')  # Remove axes
    plt.show()

    return(colored_image)





def depth_image_to_point_cloud(
    rgb, depth, intrinsic_matrix, depth_scale=1, remove_outliers=False, z_threshold=None, mask=None, device="cuda:0"
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
        # mask = mask.reshape(-1).bool()
        # mask = torch.from_numpy(mask).to(device)
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

    print(points.shape)
    print(ones.shape)
    exit()

    # remove pcd outliers
    if remove_outliers:
        scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.01)

    o3d.visualization.draw_geometries([scene_pcd])

    return np.asarray(scene_pcd.points), np.asarray(scene_pcd.colors), scene_pcd

camera_intrinsic = np.load("./orbbec_intrinsic.npy")
# print(camera_intrinsic.shape)
# exit()

video_dir = "./test_image/test30/front_rgb.npy"
depth_video = "./test_image/test30/front_depth.npy"

video = np.load(video_dir)[:5]
depth_video = np.load(depth_video)[:5]


# for image in video :
#     plt.imshow(image)
#     plt.show()
# exit()

# take a look the first video frame
# frame_idx = 0
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx}")
# plt.imshow(video[frame_idx])

inference_state = predictor.init_state(video=video)
predictor.reset_state(inference_state)


# add first object(food)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
box = np.array([200, 300, 450, 500], dtype=np.float32)
points = np.array([[260, 230]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

'''
# add second object(spoon)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[160, 170]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)


# run propagation throughout the video and collect the results in a dict
seg_list = []
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

    single_seg = torch.zeros_like(out_mask_logits[0])
    for i in out_obj_ids:
        type_seg = torch.where(out_mask_logits[i-1] > 0, torch.tensor(i), torch.tensor(0))
        single_seg += type_seg
    seg_list.append(single_seg)
all_seg = torch.cat(seg_list, dim=0)
'''

# check_seg_color(all_seg[0])

# depth_image_to_point_cloud(rgb = video[2], mask = all_seg[2], depth = depth_video[2], intrinsic_matrix = camera_intrinsic , depth_scale=1000, z_threshold=3, remove_outliers=False)




# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):

    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }



# render the segmentation results every few frames

# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(video[ann_frame_idx])

# for i, out_obj_id in enumerate(out_obj_ids):
#     show_points(points, labels, plt.gca())
#     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


pcd_list = []
vis_frame_stride = 1
# plt.close("all")
for out_frame_idx in range(0, len(video), vis_frame_stride):
    # plt.figure(figsize=(6, 4))
    # plt.title(f"frame {out_frame_idx}")
    # plt.imshow(video[out_frame_idx])
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        # show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        pcd_list.append(out_mask)
    
    _, _, _ = depth_image_to_point_cloud(rgb = video[out_frame_idx], mask = pcd_list[out_frame_idx], depth = depth_video[out_frame_idx], intrinsic_matrix = camera_intrinsic , depth_scale=1000, z_threshold=3, remove_outliers=False)
    # plt.show()


