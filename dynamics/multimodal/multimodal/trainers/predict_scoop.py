from __future__ import print_function
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from models.sensor_fusion import Dynamics_model

from dataloaders import MultimodalManipulationDataset
from torch.utils.data import DataLoader
import copy
from torch.utils.tensorboard import SummaryWriter


class predict_scoop:
    def __init__(self, configs):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
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
            training_type="scoop"
        ).to(self.device)

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)


        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs["lr"],
            betas=(self.configs["beta1"], 0.999),
            weight_decay=0.0,
        )

        

        # Weights for input
        self.alpha_vision = configs["vision"]
        self.alpha_depth = configs["depth"]
        self.alpha_eepose = configs["eepose"]


        # ------------------------
        # Handles Initialization
        # ------------------------
        if configs["load"]:
            self.load_model(configs["model_path"])

        self._init_dataloaders()

        if not os.path.exists("./../../scoop_ckpt"):
            os.makedirs("./../../scoop_ckpt")
 

    def train(self):
        train_loader = self.dataloaders["train"]
        val_loader = self.dataloaders["val"]
        best_loss = float("inf") 
        best_model_wts = None    
        self.model.train() 
        
        # TensorBoard writer
        writer = SummaryWriter(log_dir='./logs')

        for i_epoch in range(self.configs["max_epoch"]):
            print(f"epoch {i_epoch}")
            total_loss = 0.0
            total_train_acc = 0.0

            for idx, sample_batched in enumerate(tqdm(train_loader)):
                self.optimizer.zero_grad()
                loss, acc = self.loss_calc(sample_batched)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_train_acc += acc

            train_loss = total_loss / len(train_loader)
            train_acc = total_train_acc / len(train_loader)
            print(f"train_loss: {train_loss} train_accuracy: {train_acc}")

            # Log training loss and accuracy to TensorBoard
            writer.add_scalar('Loss/train', train_loss, i_epoch)
            writer.add_scalar('Accuracy/train', train_acc, i_epoch)

            val_loss = self.validate(val_loader, writer, i_epoch)
            
            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
            if (i_epoch + 1) % 10 == 0:
                FILE = f"./../../scoop_ckpt/epoch{i_epoch}.pt"
                torch.save(best_model_wts, FILE)

        writer.close()

    def validate(self, val_loader, writer, epoch):
        total_loss = 0.0
        total_acc = 0.0
        self.model.eval()

        with torch.no_grad():
            for i_iter, sample_batched in enumerate(tqdm(val_loader)):
                loss, acc = self.loss_calc(sample_batched)
                total_loss += loss.item()
                total_acc += acc

        val_loss = total_loss / len(val_loader)
        val_acc = total_acc / len(val_loader)
        
        # Log validation loss and accuracy to TensorBoard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"val_loss: {val_loss} val_accuracy: {val_acc}")
        return val_loss
    
    def load_model(self, path):
        print("Loading model from {}...".format(path))
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt)

    def loss_calc(self, sampled_batched):

        seg = sampled_batched["hand_seg"].to(self.device)
        hand_depth = sampled_batched["hand_depth"].to(self.device)
        eepose = sampled_batched["eepose"].to(self.device)
        front_depth = sampled_batched["front_depth"].to(self.device)
        # hand_pcd = sampled_batched["hand_pcd"].to(self.device)
        # front_pcd = sampled_batched[front_pcd].to(self.device)
        #label
        scoop = sampled_batched["scoop_vol"].to(self.device)
      
        
        # remove too much zero case
        reference = torch.tensor([1., 0., 0., 0., 0., 0.], device=self.device, dtype=torch.float32)
        non_zero_indices = torch.where((scoop != reference).any(dim=1))[0]
        zero_indices = torch.where((scoop == reference).all(dim=1))[0]
     
        if len(zero_indices) > int(len(non_zero_indices)/4):
            selected_indices = zero_indices[torch.randperm(len(zero_indices))[:int(len(non_zero_indices)/4)]]
            combined_indices = torch.cat((selected_indices, non_zero_indices))
            hand_depth = hand_depth[combined_indices]
            eepose = eepose[combined_indices]
            front_depth = front_depth[combined_indices]
            scoop = scoop[combined_indices]


        pred_scoop = self.model(
                hand_depth, eepose, front_depth,
            )
  
   
        loss = self.loss_function(pred_scoop, scoop)

        acc_num = 0

        for idx, pre_spillage in enumerate(pred_scoop):
            gd_class = torch.argmax(scoop[idx]).item()
            pre_class = torch.argmax(pre_spillage).item()
            if gd_class == pre_class :            
                acc_num += 1
            
        return (
            loss,
            acc_num / len(pred_scoop),
        )

    def _init_dataloaders(self):

        filename_list = []
        for file in os.listdir(self.configs["dataset"]):
            if file.endswith(".h5"):
                filename_list.append(self.configs["dataset"] + file)
  
        print(
            "Number of files in multifile dataset = {}".format(len(filename_list))
        )

        val_filename_list = []

        val_index = np.random.randint(
            0, len(filename_list), int(len(filename_list) * self.configs["val_ratio"])
        )
        

        for index in val_index:
            val_filename_list.append(filename_list[index])


        # move all val files from filename list
        while val_index.size > 0:
            filename_list.pop(val_index[0])
            val_index = np.where(val_index > val_index[0], val_index - 1, val_index)
            val_index = val_index[1:]

       
        print("Initial finished")

        self.dataloaders = {}
        self.samplers = {}
        self.datasets = {}


        self.datasets["train"] = MultimodalManipulationDataset(
            filename_list,
            data_length = self.configs["num_envs"] * self.configs["collect_time"] * self.configs["n_time_steps"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],
            single_env_steps = self.configs["collect_time"] * self.configs["n_time_steps"],
            type = "training",
        )


        self.datasets["val"] = MultimodalManipulationDataset(
            val_filename_list,
            data_length = self.configs["num_envs"] * self.configs["collect_time"] * self.configs["n_time_steps"],
            training_type=self.configs["training_type"],
            action_dim=self.configs["action_dim"],
            single_env_steps = self.configs["collect_time"] * self.configs["n_time_steps"],
            type = "validation",
        )

        print("Dataset finished")

        self.dataloaders["val"] = DataLoader(
            self.datasets["val"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.dataloaders["train"] = DataLoader(
            self.datasets["train"],
            batch_size=self.configs["batch_size"],
            num_workers=self.configs["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.len_data = len(self.dataloaders["train"])
        self.val_len_data = len(self.dataloaders["val"])

        print("Finished setting up date")

    