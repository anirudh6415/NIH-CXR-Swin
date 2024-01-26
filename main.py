from swin_transformer import SwinTransformer as st
from data import *
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"







base_path = r'../dataset_path/Images/images'
tr_label = r'../Dataset_files/Xray14_train_official.txt'
val_label = r'../Dataset_files/Xray14_val_official.txt'
ts_label = r'../Dataset_files/Xray14_test_official.txt'

train_dataset,val_dataset,test_dataset = getdataset(base_path,tr_label,val_label,ts_label)
train_loader,test_loader,val_loader = getdataloader(train_dataset,val_dataset,test_dataset,32)

model = st(in_chans = 3 , 
           num_classes = 15,
           embed_dim = 96 ,
           depths= [ 2, 2, 6, 2 ],
           num_heads= [ 3, 6, 12, 24 ],
           window_size = 7,
           drop_path_rate = 0.2
          )
model.to(device)

