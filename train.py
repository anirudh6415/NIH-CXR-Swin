import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
#from swin_transformer import SwinTransformer as st
from data import *
from sklearn.utils.class_weight import compute_class_weight
import os

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_class_weights(class_series):
    class_series = np.argmax(class_series, axis=1)
    class_unique = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_unique, y=class_series)
    return class_weights

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(epochs, model, train_loader, val_loader,device):
    #class_weights = torch.from_numpy(class_weights).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(),lr = 0.001)
    min_val_loss = np.inf
    train_losses = []
    val_losses = []
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5)
    for epoch in range(epochs):
        train_loss = 0
        print(f'Epoch: {epoch + 1}')
        model.train()
        for data in tqdm(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = 0
        model.eval()
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            out = model(images).to(device)
            loss = criterion(out, labels)
            val_loss += loss.item()
        scheduler.step(val_loss)
        lr = get_lr(optimizer)
        print(f'Epoch: {epoch + 1} \t training loss: {train_loss / len(train_loader)} \t validation loss: {val_loss / len(val_loader)} learning rate: {lr}')
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            print('Validation loss improved, saving model.')
            torch.save(model.state_dict(), 'swin_base.pth')
    return train_losses, val_losses

if __name__ == '__main__':
    swin = swinTransformer(num_classes=15)
    model_path = 'swin_base.pth'
    if os.path.exists(model_path):
        ask = input('Model exists, continue training? (y/n)')
        if ask == 'y' or ask == 'Y':
            swin.load_state_dict(model_path, device=device)
            swin.to(device)
        else:
            print('Training Fresh model ...')
            swin.to(device)
    else:
        print('Trained model does not exist. Training model from scratch ...')
        swin.to(device)
    base_path = r'CXR8\images\images_full'
    tr_label = r'CXR8\Xray14_train_official.txt'
    val_label = r'CXR8\Xray14_val_official.txt'
    ts_label = r'CXR8\Xray14_test_official.txt'
    classes = getClasses()
    train_dict, val_dict, test_dict = dataReader(base_path,tr_label, val_label, ts_label)
    print(len(train_dict), len(val_dict), len(test_dict))
    class_weights = get_class_weights(list(train_dict.values()))
    train_loader, val_loader, test_loader = getLoaders(train_dict, val_dict, test_dict)
    batch_img, batch_label = next(iter(train_loader))
    epochs = 30
    tr_loss, val_loss = train(epochs=epochs, model=swin, train_loader=train_loader, val_loader=val_loader, class_weights=class_weights)
    print(f'Training loss final: {tr_loss} Validation loss final: {val_loss}')