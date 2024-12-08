import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import json
import glob

class CroppedDataset(Dataset):
    def __init__(self, dataset, channels, transform,mask = False):
        
        self.dataset = dataset
        self.channels = channels
        self.transform = transform
        self.mask = mask
        

    def __len__(self):
        return len(self.dataset)
    
    def getCrop(self,idx):
        return self.dataset.iloc[idx]


    def __getitem__(self, idx):
        #print(idx)
        #PROCESS INPUT BANDS#################


        list_in_array = []
        for c in self.channels:
            in_raster = gdal.Open(self.dataset.iloc[idx][c])
            list_in_array.append(BandReadAsArray(in_raster.GetRasterBand(1)))
        in_array = np.array(list_in_array)
        in_tensor = self.transform(torch.as_tensor(in_array.copy().astype(np.float)))
        del in_array, list_in_array, in_raster
            
        #PROCESS LABEL BANDS#################
        label_file = self.dataset.iloc[idx]['LABEL']
        #print(label_file)
        label_raster = gdal.Open(str(label_file))
        label_array = BandReadAsArray(label_raster.GetRasterBand(1)) 
        label_tensor = torch.squeeze(torch.as_tensor(label_array.copy().astype(np.float)))
        del label_array, label_raster
        
        
        #PROCESS MASK BANDS#################
        if self.mask:
            mask_raster = gdal.Open(self.dataset.iloc[idx]['MASK'])
            mask_array = BandReadAsArray(mask_raster.GetRasterBand(1)) 
            mask_tensor = torch.squeeze(torch.as_tensor(mask_array.copy().astype(np.float)))
            label_tensor[mask_tensor == 1] = 3
            del mask_array, mask_raster
            
            
        return {
            'image': in_tensor,
            'label': label_tensor,
            'crop_idx': idx
        }

