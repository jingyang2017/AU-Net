import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_model(model, path):
    pre_trained = torch.load(path, 'cpu')
    pretrained_items = list(pre_trained.items())
    current_items = model.predictor.state_dict()
    count = 0
    for key, value in current_items.items():
        layer_name, weights = pretrained_items[count]
        current_items[key] = weights
        count = count + 1
    model.predictor.load_state_dict(current_items)
    return model
    

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

class Flatten(nn.Module):
    def forward(self,x):
        x = x.view(x.size(0),-1)
        return x

class GaussianNoise(object):
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, pic):
        arra = np.array(pic)
        noises = np.random.normal(0, self.strength, arra.shape)
        noises = np.uint8(noises)
        arra += noises
        pic = Image.fromarray(arra)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

def kl_loss(x_s, y_t, T = 4):
    p = F.log_softmax(x_s / T, dim=1)
    q = F.softmax(y_t / T, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') * (T ** 2) / x_s.shape[0]
    return l_kl

class ImageFlip(object):
    def __call__(self, img):
        img = transforms.functional.hflip(img)
        return img

def lr_change(epoch, optimizer):
    if epoch % 4 == 0:
        optimizer.param_groups[0]["lr"] *= 0.3