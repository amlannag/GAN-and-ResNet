import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

transform = transforms.Compose([
    transforms.Grayscale(),  # To ensure it's grayscale, though the dataset might already be in grayscale
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

dataset = dset.ImageFolder( root = '/Users/amlannag/Desktop/UNI/2024 Sem 2/COMP3710/keras_png_slices_data', 
                           transform= transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size= 128,
                                         shuffle=True, num_workers=2)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

print(dataset.classes)