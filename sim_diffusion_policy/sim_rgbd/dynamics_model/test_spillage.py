from __future__ import print_function
import time

import numpy as np
import torch

from .models.sensor_fusion import Dynamics_model


class spillage_predictor:
    def __init__(self):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            print("Let's use", torch.cuda.device_count(), "GPUs!")


        # model
        self.model = Dynamics_model(
            device=self.device,
        ).to(self.device)

        model_path = "./epoch15.pt"
        print("Loading model from {}...".format(model_path))
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt)


    def validate(self, eepose, tool_with_ball_pcd):
     
        # print(eepose.requires_grad)  #True 
        # print(tool_with_ball_pcd.requires_grad)   #True
        self.model.eval()
        pred_spillage = self.model(eepose, tool_with_ball_pcd)
        
        return pred_spillage
    

    

    