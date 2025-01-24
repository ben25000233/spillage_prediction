import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

np.random.seed(3)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True, color = None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        print(color)
        exit()
    else:
        color = color
   
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks,mask2, mask3 = None, scores = None, point1=None, point2 = None, point3 = None, box_coords=None, input_label_1=None, input_label_2=None, input_label_3=None, borders=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        bool_mask1 = np.array(mask, dtype=bool)
        bool_mask2 = np.array(mask2, dtype=bool)
        bool_mask3 = np.array(mask3, dtype=bool)

        # Combine the masks to get the occupied areas
        combined_mask = np.logical_or(bool_mask1, np.logical_or(bool_mask2, bool_mask3))

        # Invert the combined mask to get the unoccupied areas
        unoccupied_mask = np.logical_not(combined_mask)
        unoccupied_mask = unoccupied_mask.astype(int)
    
        red = np.array([255/255, 0/255, 0/255, 1])
        green = np.array([0/255, 255/255, 0/255, 1])
        blue = np.array([0/255, 0/255, 255/255, 1])
        yellow = np.array([255/255, 255/255, 0/255, 1])


        show_mask(mask, plt.gca(), borders=borders, color = red)
        show_mask(mask2, plt.gca(), borders=borders, color = green)
        # show_mask(mask3, plt.gca(), borders=borders, color = blue)
        # show_mask(unoccupied_mask, plt.gca(), borders=borders, color = yellow)

        # if point1 is not None:
        #     assert input_label_1 is not None
        #     show_points(point1, input_label_1, plt.gca())
        # if point2 is not None:
        #     assert input_label_2 is not None
        #     show_points(point2, input_label_2, plt.gca())
        # if point3 is not None:
        #     assert input_label_3 is not None
        #     show_points(point3, input_label_3, plt.gca())
        # if box_coords is not None:
        #     # boxes
        #     show_box(box_coords, plt.gca())
        # if len(scores) > 1:
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def vis_mask(image):
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=bowl_point,
        point_labels=bowl_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]

    masks2, scores, logits = predictor.predict(
        point_coords=ball_point,
        point_labels=ball_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks2 = masks2[sorted_ind]
    
    '''
    masks3, scores, logits = predictor.predict(
        point_coords=tabel_point,
        point_labels=tabel_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks3 = masks3[sorted_ind]

    masks4, scores, logits = predictor.predict(
        point_coords=scoop_point,
        point_labels=scoop_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks4 = masks4[sorted_ind]
    '''

    show_masks(image, masks, masks2, point1=bowl_point, point2=ball_point, input_label_1=bowl_label, input_label_2=ball_label)
    # show_masks(image, masks, masks2, masks3, point1=bowl_point, point2=ball_point, point3=tabel_point, input_label_1=bowl_label, input_label_2=ball_label, input_label_3=tabel_label)
    






sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

predictor = SAM2ImagePredictor(sam2)


# image = plt.imread('test_image/real.png')
# image = image[:, :, :3]

# masks = mask_generator.generate(image)

# plt.imshow(image)
# show_anns(masks)
# plt.show()
# plt.savefig('real_seg.png')
# exit()


# input_point = np.array([[[90, 77], [80, 120]], [[90, 90]]])
bowl_point = np.array([[63, 91]])
# input_label = np.array([[1, 1], [1]])
bowl_label = np.array([1])

ball_point = np.array([[85, 80]])
ball_label = np.array([1])


# tabel_point = np.array([[10, 100]])
# tabel_label = np.array([1])

# scoop_point = np.array([[50, 30]])
# scoop_label = np.array([1])

import cv2
images = np.load('test_image/rgb_front.npy')


for i in range(80, 200, 1):
    image = images[i]
    image = cv2.resize(image, (320, 240))
    image = image[30:135, 50:165]

    vis_mask(image)

    #auto generate
    # masks = mask_generator.generate(image)

    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 

    
