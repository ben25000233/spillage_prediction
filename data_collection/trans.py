import numpy as np
import cv2

def project_world_point_to_image(world_point, extrinsic, intrinsic):
    
    # Convert world point to camera coordinates using the extrinsic matrix
    camera_coords = extrinsic[:3, :3].dot(world_point) + extrinsic[:3, 3]
    
    # Project the camera coordinates to image plane using intrinsic matrix
    image_coords = intrinsic.dot(camera_coords)
    print(image_coords)
    exit()
    
    # Normalize the coordinates to get pixel coordinates (divide by z)
    pixel_coords = image_coords[:2] / image_coords[2]

    print(pixel_coords)
    exit()
    
    return pixel_coords

def visualize_point_on_image(image, pixel_coords):
    # Convert pixel coordinates to integer values
    pixel_coords = np.round(pixel_coords).astype(int)
    
    # Draw a circle at the pixel coordinates
    cv2.circle(image, (pixel_coords[0], pixel_coords[1]), radius=5, color=(0, 0, 255), thickness=-1)
    
    # Display the image with the point visualized
    cv2.imshow("Image with 3D Point", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_camera_intrinsics_matrix(image_width, image_heigth, horizontal_fov):
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180
    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
    K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]])
    return K

# Example usage (using Matplotlib for visualization)



# Example usage
world_point = np.array([0, 0, 0])  # A 3D world point
quaternion = np.array([1, 0, 0, 0])  # Example quaternion

intrinsic_matrix = compute_camera_intrinsics_matrix(1280, 960, 65)

extrinsic = np.load("./real_cam_pose/back_cam2base.npy")


# Load your image (ensure it's in the correct format, e.g., BGR for OpenCV)
image = cv2.imread("back_rgb_image.jpg")  # Replace with your image path


# Project the 3D point onto the image using the extrinsic and intrinsic matrices
pixel_coords = project_world_point_to_image(world_point, extrinsic, intrinsic_matrix)

# Visualize the point on the image
visualize_point_on_image(image, pixel_coords)
