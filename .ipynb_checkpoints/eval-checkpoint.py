import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from data import *
import os



def test(model,test_loader,device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    test_losses = []
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for data in tqdm(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            out = model(images).to(device)
            loss = criterion(out, labels)
            test_loss += loss.item()
        print(f'testing loss: {test_loss / len(test_loader)}')
    return out,test_loss/ len(test_loader)
