import cv2
import numpy as np
import torch
import sys
import os

from torch.nn import MSELoss

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.models.cnn_vae import CnnVae

def kl_divergence(x, y, mu, log_var):
    criterion = MSELoss()

    bce_loss = criterion(x, y)

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return bce_loss + kld

def initialize_model(in_channels, model_type, device):
    model = None

    if model_type == 'cnn_vae':
        model = CnnVae(
            in_channels=in_channels
        ).to(device)
    else:
        raise ValueError(f'Unsupported model type {model_type}')

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
