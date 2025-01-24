import numpy as np
import cv2
import matplotlib.pyplot as plt

video_dir = "./test_image/top_obs.npy"
video = np.load(video_dir)

for i in range(len(video)):
    image = video[i]
    plt.imshow(image)
    plt.show()
    exit()
