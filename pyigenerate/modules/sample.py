import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pyigenerate.lib.utils import initialize_model

class Sample:
    def __init__(self, params={}):
        self.params = params

        self.img_width  = params.get('img_width')
        self.img_height = params.get('img_height')
        self.device     = params.get('device')
        self.gpu_index  = params.get('gpu_index') or 0
        self.model_file = params.get('model_file')
        self.model_type = params.get('model_type')
        self.latent_dim = params.get('latent_dim')

    def execute(self):
        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file)

        model = initialize_model(3, self.model_type, self.device)

        model.load_state_dict(state['state_dict'])

        sampled_mu = torch.Tensor([np.zeros(self.latent_dim)]).to(self.device)
        sampled_log_var = torch.Tensor([np.zeros(self.latent_dim)]).to(self.device)

        result = model.sample(sampled_mu, sampled_log_var)[0].detach().cpu().numpy()
        print(result.shape)
        result = result.transpose((1, 2, 0))
        print(result.shape)
        plt.imshow(result)
        plt.show()
