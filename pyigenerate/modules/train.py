import sys
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.utils import initialize_model

class Train:
    def __init__(self, params={}, seed=0):
        if seed >= 0:
            torch.manual_seed(seed)

        self.params = params

        self.img_width              = params.get('img_height')
        self.img_height             = params.get('img_height')
        self.device                 = params.get('device')
        self.gpu_index              = params.get('gpu_index')
        self.input_img_dir          = params.get('input_img_dir')
        self.epochs                 = params.get('epochs')
        self.model_file             = params.get('model_file')
        self.learning_rate          = params.get('learning_rate')
        self.batch_size             = params.get('batch_size')
        self.in_channels            = params.get('in_channels') or 3
        self.cont                   = params.get('cont') or False
        self.loss_type              = params.get('loss_type') or 'CE'
        self.model_type             = params.get('model_type') or 'cnn-vae'

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
