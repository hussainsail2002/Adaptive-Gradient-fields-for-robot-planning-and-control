import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import torch
import torch.nn as nn
import csv
import pandas as pd
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from meshcat.animation import Animation
import meshcat.transformations as tf
import meshcat
from meshcat.geometry import Box
import meshcat.geometry as g
import time
import math
import pyautogui
from casadi import *

num_neuron = 35

class MyNueralNetworkA(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,4))
        
    def forward(self,x1):
        return self.network(x1).squeeze()

    
class MyNueralNetworkB(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,num_neuron),
            nn.ReLU(),
            nn.Linear(num_neuron,4))
    
    def forward(self,x1):
        return self.network(x1).squeeze()