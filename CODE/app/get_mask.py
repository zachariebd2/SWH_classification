import pandas as pd
import os
import numpy as np
import rasterio
import glob
import argparse


#PARSE INPUT ARGUMENTS
print("PARSE INPUT ARGUMENTS")
arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-c', action='store', default="", nargs='?', const='', dest='csv_file')
csv_file = arg_parser.parse_args().csv_file

CLOUD_VALUE = 205
BIT_MONOTEMP_CLOUDS = 2

df = pd.read_csv(csv_file, dtype = str)
for index, row in df.iterrows():
    print(os.path.join((row['L2A_PATH'])[:-7]+'*','MASKS',"*_CLM_R2.tif"))
    L2B_raster_path = glob.glob(os.path.join(row['L2B_PATH'],"*","*_SNW_R2.tif"))[0]
    CLM_raster_path = glob.glob(os.path.join((row['L2A_PATH'])[:-7]+'*','MASKS',"*_CLM_R2.tif"))[0]
    
    with rasterio.open(L2B_raster_path) as src:
        meta = src.meta.copy()
        L2B = src.read(1)
    CLM = (rasterio.open(CLM_raster_path)).read(1) 
    WTR = (rasterio.open(row['WATER_PATH'])).read(1) 
    TCD = (rasterio.open(row['TCD_PATH'])).read(1) 

    MASK = np.where((TCD > 50)  | (WTR == 1) |
                    ((L2B == CLOUD_VALUE ) &
                     (np.bitwise_and(CLM,2**BIT_MONOTEMP_CLOUDS) != 2**BIT_MONOTEMP_CLOUDS)) ,1,0)
    os.system(f"mkdir -p {os.path.dirname(row['MASK_PATH'])}")
    with rasterio.open(row['MASK_PATH'], "w", **meta) as ds:
        ds.write(MASK,1)
