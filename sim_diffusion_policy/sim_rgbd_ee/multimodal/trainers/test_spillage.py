from __future__ import print_function
import numpy as np
import torch
from models.sensor_fusion import Dynamics_model


class Test_dynamics:
    def __init__(self, configs):

        use_cuda = configs["cuda"] and torch.cuda.is_available()

        self.configs = configs
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        # model
        self.model = Dynamics_model(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
            action_dim=configs["action_dim"],
        ).to(self.device)

        print("Loading model from {}...".format(configs["model_path"]))
        ckpt = torch.load(configs["model_path"])
        self.model.load_state_dict(ckpt)


        

    def test(self, eepose, obsvation):

        with torch.no_grad():
            pred_spillage = self.model(obsvation, eepose)
            

   


        

    