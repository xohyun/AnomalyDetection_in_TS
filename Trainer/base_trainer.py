from abc import *
import torch.nn as nn
import torch
import torch.optim as optim
from utils.utils import gpu_checking
from abc import ABC

class base_trainer(ABC):  
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluation(self):
        pass

    @abstractmethod
    def set_criterion(self, criterion):
        pass

    @abstractmethod
    def set_scheduler(self, args, optimizer):
        pass