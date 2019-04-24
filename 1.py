import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image, ImageOps

def channel_1toN(img, num_channel):
    transform1 = transforms.Compose([transforms.ToTensor(),])
    img = (transform1(img) * 255.0).long()
    T = torch.LongTensor(num_channel, img.size(1), img.size(2)).zero_()
    #N = (torch.rand(num_channel, img.size(1), img.size(2)) - 0.5)/random.uniform(1e10, 1e25)#Noise
    mask = torch.LongTensor(img.size(1), img.size(2)).zero_()
    for i in range(num_channel):
        T[i] = T[i] + i
        layer = T[i] - img
        T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
    return T.float()

def channel_1to1(img):
    transform1 = transforms.Compose([transforms.ToTensor(),])
    T = torch.LongTensor(img.height, img.width).zero_()
    img = (transform1(img) * 255.0).long()
    print(img.shape)
    T.resize_(img[0].size()).copy_(img[0])
    print(T)
    return T.long()
path='1.png'
A=Image.open(path)
A=A.resize((3,4))
# print(A.long())
A1=channel_1to1(A)
# print(A1)
AN=channel_1toN(A,15)
print(AN)