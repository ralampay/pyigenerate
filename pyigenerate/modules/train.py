import sys
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.custom_dataset import CustomDataset
from lib.utils import initialize_model, kl_divergence

class Train:
    def __init__(self, params={}, seed=0):
        if seed >= 0:
            torch.manual_seed(seed)

        self.params = params

        self.img_width              = params.get('img_width')
        self.img_height             = params.get('img_height')
        self.device                 = params.get('device')
        self.gpu_index              = params.get('gpu_index')
        self.input_img_dir          = params.get('input_img_dir')
        self.epochs                 = params.get('epochs')
        self.model_file             = params.get('model_file')
        self.learning_rate          = params.get('learning_rate')
        self.batch_size             = params.get('batch_size') or 2
        self.in_channels            = params.get('in_channels') or 3
        self.cont                   = params.get('cont') or False
        self.loss_type              = params.get('loss_type') or 'KLD'
        self.model_type             = params.get('model_type') or 'cnn_vae'
        self.model_file             = params.get('model_file')

        self.model = None

    def execute(self):
        print(f"Training model {self.model_type}...")

        print("input_img_dir: {}".format(self.input_img_dir))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        self.model = initialize_model(
            self.in_channels,
            self.model_type,
            self.device
        )

        print(self.model)

        if self.cont and os.path.exists(self.model_file):
            state = torch.load(
                self.model_file, 
                map_location=self.device
            )

            self.model.load_state_dict(state['state_dict'])
            self.model.optimizer     = state['optimizer']

        optimizer   = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scaler      = torch.cuda.amp.GradScaler()

        train_ds = CustomDataset(
            image_dir=self.input_img_dir,
            img_width=self.img_width,
            img_height=self.img_height
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch+1))

            ave_loss = self.train_fn(
                train_loader,
                self.model,
                optimizer,
                scaler
            )

            print("Ave Loss: {}".format(ave_loss))

            # Save model after every epoch
            print("Saving model to {}...".format(self.model_file))

            state = {
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(state, self.model_file)

    def train_fn(self, loader, model, optimizer, scaler):
        loop = tqdm(loader)

        ave_loss = 0.0
        count = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data    = data.float().to(device=self.device)
            targets = targets.to(device=self.device)

            # Forward
            predictions, mu, log_var = model.forward(data)

            #loss = loss_fn(predictions, targets)
            loss = kl_divergence(predictions, targets, mu, log_var)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm
            loop.set_postfix(loss=loss.item())

            # Write to tensorboard

            ave_loss += loss.item()
            count += 1

        ave_loss = ave_loss / count

        return ave_loss

