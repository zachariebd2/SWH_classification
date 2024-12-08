
import matplotlib.pyplot as plt
import os
from data_loading import CroppedDataset
from EarlyStopper import EarlyStopper
from evaluate import evaluate
from unet import UNet
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch import optim
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import seaborn as sns
import argparse
from copy import deepcopy
from utils import mkdir_p, makeCropDataset
from pathlib import Path
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
import glob
import json



#torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#torch.use_deterministic_algorithms()



#PARSE INPUT ARGUMENTS
print("PARSE INPUT ARGUMENTS")
arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-c', action='store', default="", nargs='?', const='', dest='config_file')
arg_parser.add_argument('-tmp', action='store', default="", nargs='?', const='', dest='tmp_dir')
config_file = arg_parser.parse_args().config_file
tmp_dir = arg_parser.parse_args().tmp_dir


#ACCESS CONFIGURATION FILE
print("ACCESS CONFIGURATION FILE")
with open(config_file, "r") as f:
    config = json.load(f) 
input_csv = config["input_csv"]
sat = config["sat"]
model = config["model"]
channels = config["channels"]
draws = config["draws"]
tiles = config["tiles"]
work_dir = config["work_dir"]
epochs = int(config["epochs"])
learning_rate = float(config["learning_rate"])
patience = int(config["patience"])
area = config["area"]




#GET DATAFRAME
print("GET DATAFRAME")
df = pd.read_csv(input_csv)
train_split = df.query(f"PSEUDO_SATELLITE == '{sat}'  & tile in @tiles & SET == 'TRAIN' & draw <= {draws} ")
val_split  = df.query(f"PSEUDO_SATELLITE == '{sat}'  & tile in @tiles  & SET == 'TEST' & draw <= {draws} ")

nb_channels = len(channels)


#GET WEIGHTS AND NORMALIZATION PARAMS
init_path = os.path.join(work_dir,"MODELS",model,area,sat,"INIT")
train_weights = torch.load(os.path.join(init_path,'weights_from_train_set.pth')).to(device=device, dtype=torch.float32)
val_weights = torch.load(os.path.join(init_path,'weights_from_val_set.pth')).to(device=device, dtype=torch.float32)
train_norm = torch.load(os.path.join(init_path,'normalization_from_train_set.pth')).to(device=device, dtype=torch.float32)
print("train weights",train_weights)
print("val weights",val_weights)
print("train_norm",train_norm)


#SET DATALOADERS
print('set dataloaders')
train_set = CroppedDataset(train_split,channels,train_norm,mask=True)
val_set = CroppedDataset(val_split,channels,train_norm,mask=True)
loader_args = dict(batch_size=11, num_workers=8, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
print("len train loader",len(train_loader))
print("len val loader",len(val_loader))
nb_classes = len(train_weights)






#TRAIN UNET FROM CHECKPOINT
out_path = os.path.join(work_dir,"MODELS",model,area,sat,"STEP2")
os.system(f"mkdir -p {out_path}")
early_stopper = EarlyStopper(patience=patience)
print("UNET setup")
checkpoint_path = os.path.join(work_dir,"MODELS",model,area,sat,"CHECKPOINT")
net = torch.load(glob.glob(os.path.join(checkpoint_path,'*model*'))[0])
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
net.to(device)
best_opti = optimizer.state_dict()
best_grad = grad_scaler.state_dict()
best_net_model = deepcopy(net)
#loss setup
print("loss setup")
train_criterion = nn.CrossEntropyLoss(weight = train_weights,reduction='mean', ignore_index=3)
val_criterion = nn.CrossEntropyLoss(weight = val_weights,reduction='mean', ignore_index=3)
print("begin training")
net.train()
checkpoint = 1
best_loss = 10000
best_checkpoint = 0
train_tile_nb = len(train_loader.dataset) 
val_tile_nb = len(val_loader.dataset) 
train_btch_nb = len(train_loader)
val_btch_nb = len(val_loader)
df_pre_train = pd.DataFrame(columns=['score_value','score_type','checkpoint'])

x_margin,y_margin = net.getOutput2DMargins()
print(x_margin,y_margin)
for epoch in range(1,epochs+1):

    print("epoch",epoch)
    train_loss = 0

    #training
    for batch_idx , batch in enumerate(train_loader):
        #training round
        images = batch['image']
        print("image size",images.size())
        print("label size",batch['label'].size())
        true_labels = batch['label'][:,x_margin:-1*x_margin,y_margin:-1*y_margin]
        images = images.to(device=device, dtype=torch.float32)
        true_labels = true_labels.to(device=device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=False):
            pred_probs = net(images)
            pred_probs = pred_probs.to(device=device, dtype=torch.float32)
            #print("pred size",pred_probs.size())
            loss = train_criterion(pred_probs, true_labels)
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward() 
        grad_scaler.step(optimizer)
        grad_scaler.update()
        train_loss += loss.item()

    train_loss = train_loss / train_btch_nb

    #testing
    val_loss , dice_score = evaluate(net, val_loader, device, val_criterion)

    score_list = [pd.Series([val_loss, 'val_loss', checkpoint], index=df_pre_train.columns ) ,
                  pd.Series([train_loss, 'train_loss', checkpoint], index=df_pre_train.columns ) ]
    df_pre_train = df_pre_train.append(score_list, ignore_index=True)


    common_name = f'step_2'
    df_plot = df_pre_train


    fig = plt.figure()
    plt.title('performances {} epoch {}'.format(common_name,best_checkpoint)) 
    sns.lineplot(x='checkpoint', y="score_value", hue = 'score_type',data=df_plot)
    plt.savefig(os.path.join(out_path,f'performances_{common_name}.png'))
    plt.close()
    df_pre_train.to_csv(out_path+'/checkpoints_{}.csv'.format(common_name), index=False)     
    #mloss = max(val_loss,float(loss))
    if val_loss < best_loss:
        best_loss = val_loss
        best_opti = optimizer.state_dict()
        best_net = net.state_dict()
        best_grad = grad_scaler.state_dict()
        best_checkpoint = checkpoint
        best_net_model = deepcopy(net)
        torch.save(best_net_model, out_path+f'/checkpoint_model_{common_name}_epoch_{best_checkpoint}.pkl')
        torch.save(best_net, out_path+f'/checkpoint_state_{common_name}_epoch_{best_checkpoint}.pth')
        torch.save(best_opti, out_path+f'/checkpoint_optim_{common_name}_epoch_{best_checkpoint}.pth')
        torch.save(best_grad, out_path+f'/checkpoint_grad_{common_name}_epoch_{best_checkpoint}.pth')
    checkpoint += 1
    if early_stopper.early_stop(val_loss): break  

