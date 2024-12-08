
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
labels = config["labels"]
mask = config["mask"]
tiles = config["tiles"]
area = config["area"]
work_dir = config["work_dir"]

#GET DATAFRAME
print("GET DATAFRAME")
df = pd.read_csv(input_csv)
train_split = df.query(f"PSEUDO_SATELLITE == '{sat}'  & tile in @tiles & SET == 'TRAIN' & draw <= {draws} ")
test_split  = df.query(f"PSEUDO_SATELLITE == '{sat}'  & tile in @tiles  & SET == 'TEST' & draw <= {draws} ")

nb_channels = len(channels)

#GET UNET DIFFERENCE MARGINS BETWEEN INPUT AND OUTPUT
net = UNet(n_channels=0, n_classes=0)
x_margin,y_margin = net.getOutput2DMargins()
print(x_margin,y_margin)





#NORMALIZATION FROM TRAINING SET
print("normalization from training set")
means = []
stds = []
for c in channels:
    list_r = []
    for index, row in df.iterrows():
        r_raster = gdal.Open(str(row[c]))
        r_array = BandReadAsArray(r_raster.GetRasterBand(1))
        list_r.append(r_array)
    array_r = np.concatenate(list_r)
    means.append(np.mean(array_r))
    stds.append(np.std(array_r))


train_norm = torch.nn.Sequential(
        transforms.Normalize(means, stds)
    )




train_set = CroppedDataset(train_split,channels,train_norm,mask=False)
val_set = CroppedDataset(test_split,channels,train_norm,mask=False)


#training weigths
print("training weigths")
list_label_tensors = []


for train_id in range(len(train_set)):
    label_tensor = train_set[train_id]['label'][x_margin:-1*x_margin,y_margin:-1*y_margin]
    list_label_tensors.append(label_tensor)
label_tensor = torch.cat(list_label_tensors)
unique_values, counts = torch.unique(label_tensor, return_counts=True) 
print(unique_values,unique_values.size(dim=0),counts,torch.sum(counts))
train_weights = torch.sum(counts[unique_values!=3])/(counts)
train_weights = train_weights.to(device=device, dtype=torch.float32)
print(train_weights)
nb_classes = len(unique_values)



#validation weigths
print("validation weigths")
list_label_tensors = []


for val_id in range(len(val_set)):
    label_tensor = val_set[val_id]['label'][x_margin:-1*x_margin,y_margin:-1*y_margin]
    list_label_tensors.append(label_tensor)
label_tensor = torch.cat(list_label_tensors)
unique_values, counts = torch.unique(label_tensor, return_counts=True) 
print(unique_values,unique_values.size(dim=0),counts,torch.sum(counts))
val_weights = torch.sum(counts[unique_values!=3])/(counts)
val_weights = val_weights.to(device=device, dtype=torch.float32)
print(val_weights)
nb_classes = len(unique_values)


out_path = os.path.join(work_dir,"MODELS",model,area,sat,"INIT")
os.system(f"mkdir -p {out_path}")


torch.save(train_norm, out_path+f'/normalization_from_train_set.pth')
torch.save(train_weights, out_path+f'/weights_from_train_set.pth')
torch.save(val_weights, out_path+f'/weights_from_val_set.pth')




