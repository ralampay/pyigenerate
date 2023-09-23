from math import e
import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pyigenerate.lib.utils import initialize_model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.custom_dataset import CustomDataset

class Sample:
    def __init__(self, params={}):
        self.params = params

        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.device         = params.get('device')
        self.gpu_index      = params.get('gpu_index') or 0
        self.model_file     = params.get('model_file')
        self.model_type     = params.get('model_type')
        self.latent_dim     = params.get('latent_dim')
        self.input_img_dir  = params.get('input_img_dir')


        self.dataset = CustomDataset(
            image_dir=self.input_img_dir,
            img_width=self.img_width,
            img_height=self.img_height
        )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True
        )

    def execute(self):
        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file)

        model = initialize_model(3, self.model_type, self.device)

        model.load_state_dict(state['state_dict'])

        model.eval()

        while True:
            data_iter = iter(self.data_loader)
            sample_x, _ = next(data_iter)

            _, mu, log_var = model.encode(sample_x.to(self.device))

            print(mu)
            print(log_var)

            result = model.sample(mu, log_var)[0].detach().cpu().numpy()
            result = result.transpose((1, 2, 0))
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            result = cv2.resize(result, (250, 250))

            cv2.imshow('sample', result)

            key = cv2.waitKey(1) & 0xff

            if key == 27:
                break
