import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from data_loading import CroppedDataset
import pandas as pd
from unet import UNet
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch import optim
import torch.nn as nn
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import argparse
import pickle
import errno
import json
from utils import mkdir_p, makeCropDataset
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)






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
area = config["area"]



#GET INPUT DATAFRAME AND CREATE OUTPUT DATAFRAME
print("GET DATAFRAME")
df = pd.read_csv(input_csv, dtype = str)
df['draw'] = df['draw'].astype(int)
eval_split = df.query(f"PSEUDO_SATELLITE == '{sat}'  & tile in @tiles & SET == 'EVAL' & draw <= {draws} ").astype(str)
nb_channels = len(channels)


eval_split["PREDICTION"] = f"{work_dir}/MODELS/{model}/{area}/{sat}/EVAL/"+eval_split['tile']+"/"+eval_split['year']+"/"+eval_split['month']+"/"+eval_split['day']+"/"+"PREDICTED_"+sat+"_"+eval_split['tile']+"_"+eval_split['year']+eval_split['month']+eval_split['day']+"_"+eval_split['row']+"_"+eval_split['col']+"_DRAW_"+eval_split['draw']+".tif"

eval_split["PROB_GROUND"] = f"{work_dir}/MODELS/{model}/{area}/{sat}/EVAL/"+eval_split['tile']+"/"+eval_split['year']+"/"+eval_split['month']+"/"+eval_split['day']+"/"+"PROB0_"+sat+"_"+eval_split['tile']+"_"+eval_split['year']+eval_split['month']+eval_split['day']+"_"+eval_split['row']+"_"+eval_split['col']+"_DRAW_"+eval_split['draw']+".tif"

eval_split["PROB_SNOW"] = f"{work_dir}/MODELS/{model}/{area}/{sat}/EVAL/"+eval_split['tile']+"/"+eval_split['year']+"/"+eval_split['month']+"/"+eval_split['day']+"/"+"PROB1_"+sat+"_"+eval_split['tile']+"_"+eval_split['year']+eval_split['month']+eval_split['day']+"_"+eval_split['row']+"_"+eval_split['col']+"_DRAW_"+eval_split['draw']+".tif"

eval_split["PROB_CLOUD"] = f"{work_dir}/MODELS/{model}/{area}/{sat}/EVAL/"+eval_split['tile']+"/"+eval_split['year']+"/"+eval_split['month']+"/"+eval_split['day']+"/"+"PROB2_"+sat+"_"+eval_split['tile']+"_"+eval_split['year']+eval_split['month']+eval_split['day']+"_"+eval_split['row']+"_"+eval_split['col']+"_DRAW_"+eval_split['draw']+".tif"

out_csvs = os.path.join(work_dir,"MODELS",model,area,sat,"CSVS")
os.system(f"mkdir -p {out_csvs}")
eval_split.to_csv(os.path.join(out_csvs,f"PATCHES.csv"), index=False)

#LOAD MODEL
print('load model')
checkpoints_path = os.path.join(work_dir,"MODELS",model,area,sat,"STEP2","checkpoints_step_2.csv")
df_checkpoints = pd.read_csv(checkpoints_path)
best_checkpoint = df_checkpoints.loc[df_checkpoints.query("score_type == 'val_loss'")['score_value'].idxmin(), 'checkpoint']
model_path = os.path.join(work_dir,"MODELS",model,area,sat,"STEP2",f"checkpoint_model_step_2_epoch_{best_checkpoint}.pkl")
net = torch.load(model_path).to(device=device) 
net.eval()

#ADD INPUT-OUTPUT MARGINS IN DATAFRAME
x_margin,y_margin = net.getOutput2DMargins()
eval_split["XMARGIN"] = x_margin
eval_split["YMARGIN"] = y_margin


#GET WEIGHTS AND NORMALIZATION PARAMS
init_path = os.path.join(work_dir,"MODELS",model,area,sat,"INIT")
train_norm = torch.load(os.path.join(init_path,'normalization_from_train_set.pth')).to(device=device, dtype=torch.float32)
print("train_norm",train_norm)


#SET DATALOADERS
print('set dataloaders')
eval_set = CroppedDataset(eval_split,channels,train_norm,mask=True)
loader_args = dict(batch_size=11, num_workers=8, pin_memory=True)
eval_loader = DataLoader(eval_set, shuffle=False, **loader_args)
print("len eval loader",len(eval_loader))




driver = gdal.GetDriverByName('GTiff')
output_path = os.path.join(work_dir,"MODELS",model,area,sat,"EVAL")
os.system(f"mkdir -p {output_path}")

m = nn.Softmax(dim=1)
for batch_idx , batch in enumerate(eval_loader):
    #get batch
    print("batch",batch_idx)
    images = batch['image'] # (batches,bands,x,y)
    true_labels = batch['label'][:,x_margin:-1*x_margin,y_margin:-1*y_margin] # (batches,x,y)
    crops_idx = batch['crop_idx'] #(batches)
    print(crops_idx)
    images = images.to(device=device, dtype=torch.float32)
    true_labels = true_labels.to(device=device, dtype=torch.long) 
    #calculate pred and loss
    with torch.no_grad():
        prob_labels = net(images)  # (batches,classes,x,y)
        prob_labels = prob_labels.to(device=device, dtype=torch.float32)
        pred_labels = prob_labels.argmax(dim=1) # (batches,x,y)   recuperer avant argmax
        prob_labels = m(prob_labels) # (batches,classes,x,y)
        
    #for each crop in batch
    for crop_nb in range(crops_idx.size(dim=0)):
        print("crop_nb",crop_nb)
        #get crop
        print("crop_idx",crops_idx[crop_nb])
        print("crop_item",crops_idx[crop_nb].item())
        crop = eval_set.getCrop(crops_idx[crop_nb].item())
        print("crop",crop['tile'])
        #create output dir
        crop_out = os.path.dirname(crop["PREDICTION"])
        os.system(f"mkdir -p {crop_out}")
        
        print(crop['LABEL'])
        
        print(crop["PROB_GROUND"])
        print(crop["PREDICTION"])
        
        
       

        #create prob tiff (using label raster for the geo)
        raster = gdal.Open(crop['LABEL'])
        in_xsize = raster.RasterXSize
        in_ysize = raster.RasterYSize
        del raster
        raster = gdal.Translate("",crop['LABEL'],format="MEM",srcWin = [x_margin, y_margin,in_xsize - 2*x_margin, in_ysize - 2*y_margin ])
        prob_array = prob_labels[crop_nb,0,...].cpu().detach().numpy() 
        outDs = driver.Create(crop["PROB_GROUND"],  raster.RasterXSize, raster.RasterYSize , 1, gdal.GDT_Float32)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(prob_array)
        outDs.SetGeoTransform(raster.GetGeoTransform()) 
        outDs.SetProjection(raster.GetProjection())
        outDs.FlushCache()
        
        prob_array = prob_labels[crop_nb,1,...].cpu().detach().numpy() 
        outDs = driver.Create(crop["PROB_SNOW"],  raster.RasterXSize, raster.RasterYSize , 1, gdal.GDT_Float32)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(prob_array)
        outDs.SetGeoTransform(raster.GetGeoTransform()) 
        outDs.SetProjection(raster.GetProjection())
        outDs.FlushCache()
        

        prob_array = prob_labels[crop_nb,2,...].cpu().detach().numpy() 
        outDs = driver.Create(crop["PROB_CLOUD"],  raster.RasterXSize, raster.RasterYSize , 1, gdal.GDT_Float32)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(prob_array)
        outDs.SetGeoTransform(raster.GetGeoTransform()) 
        outDs.SetProjection(raster.GetProjection())
        outDs.FlushCache()
        
        
        
        #create pred tiff (using label raster for the geo)
        pred_array = pred_labels[crop_nb,...].cpu().detach().numpy()     
        outDs = driver.Create(crop["PREDICTION"],  raster.RasterXSize, raster.RasterYSize , 1, gdal.GDT_Byte)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(pred_array)
        outDs.SetGeoTransform(raster.GetGeoTransform()) 
        outDs.SetProjection(raster.GetProjection())
        outDs.FlushCache()      
        
        
        

