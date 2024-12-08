import os
import glob
import json
import argparse
import numpy as np
import cloudpickle
import pandas as pd
from datetime import datetime, timedelta, date
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import copy



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
output_dir = config["project_dir"]
patch_size = config["patch_size"]
overlap = config["overlap"]
res = config["res"]
index = config["index"]

#GET DATAFRAME
print("GET DATAFRAME")
df = pd.read_csv(input_csv, dtype = str)



#MAKE PATCHES

list_df = []
#for index, row in df.iterrows():
row = df.iloc[index]
out_path = os.path.join(output_dir,"TRAINING","S2","PATCHES",row['tile'],row['year'],row['month'],row['day'])



os.system(f"mkdir -p {out_path}")


#S2L1C
for band in ['B02','B03','B04','B08','B11']:
    in_path = glob.glob(os.path.join(row['L1C_PATH'],"GRANULE",'*','IMG_DATA',f"*_{band}.jp2"))[0]
    raster= gdal.Warp("",in_path,format="MEM",overviewLevel = None) 
    ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
    tmp_file = tmp_dir+f"/{row['tile']}_{row['year']+row['month']+row['day']}_L1C_{band}.tif"
    csv_path = f"temp.csv"
    if abs(xres) < res:
        resampleAlg = "bilinear"
    elif abs(xres) > res:
        resampleAlg = "cubic"
    else:
        resampleAlg = "nearest"
    
    gdal.Warp(tmp_file,raster,format= 'GTiff',resampleAlg=resampleAlg,xRes= res,yRes= res,creationOptions = ['COMPRESS=DEFLATE',"PREDICTOR=2"])

    os.system(f"gdal_retile.py -ps {patch_size} {patch_size} -overlap {overlap} -tileIndexField patch -csv {csv_path} -targetDir {out_path} {tmp_file}")
    df_patch = pd.read_csv(os.path.join(out_path,csv_path), dtype = str,sep=';',names=["tilename","minx","maxx","miny","maxy"])
    df_patch["tile"] = row['tile']
    df_patch["area"] = row['area']
    df_patch["INSTRUMENT"] = row['INSTRUMENT']
    df_patch["SATELLITE"] = row['SATELLITE']
    df_patch["year"] = row['year']
    df_patch["month"] = row['month']
    df_patch["day"] = row['day']
    df_patch["band"] = band
    df_patch["type"] = "L1C"
    df_patch["row"] = df_patch.tilename.str[-9:-7]
    df_patch["col"] = df_patch.tilename.str[-6:-4]
    df_patch["path"] = out_path+"/"+df_patch["tilename"]
    list_df.append(df_patch)
    os.system(f"rm {os.path.join(out_path,csv_path)}")
    
    
#ELEVATION
raster= gdal.Open(row['DEM_PATH']) 
ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
tmp_file = tmp_dir+f"/{row['tile']}_{row['year']+row['month']+row['day']}_DEM.tif"
csv_path = f"temp.csv"
if abs(xres) < res:
    resampleAlg = "bilinear"
elif abs(xres) > res:
    resampleAlg = "cubic"
else:
    resampleAlg = "nearest"
gdal.Warp(tmp_file,raster,format= 'GTiff',resampleAlg=resampleAlg,xRes= res,yRes= res,creationOptions = ['COMPRESS=DEFLATE',"PREDICTOR=2"])
os.system(f"gdal_retile.py -ps {patch_size} {patch_size} -overlap {overlap} -tileIndexField patch -csv {csv_path} -targetDir {out_path} {tmp_file}")
df_patch = pd.read_csv(os.path.join(out_path,csv_path), dtype = str,sep=';',names=["tilename","minx","maxx","miny","maxy"])
df_patch["tile"] = row['tile']
df_patch["area"] = row['area']
df_patch["INSTRUMENT"] = row['INSTRUMENT']
df_patch["SATELLITE"] = row['SATELLITE']
df_patch["year"] = row['year']
df_patch["month"] = row['month']
df_patch["day"] = row['day']
df_patch["band"] = "DEM"
df_patch["type"] = "DEM"
df_patch["row"] = df_patch.tilename.str[-9:-7]
df_patch["col"] = df_patch.tilename.str[-6:-4]
df_patch["path"] = out_path+"/"+df_patch["tilename"]
list_df.append(df_patch)
os.system(f"rm {os.path.join(out_path,csv_path)}")
    
    
#HILLSHADE
raster= gdal.Open(row['HILL_PATH']) 
ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
tmp_file = tmp_dir+f"/{row['tile']}_{row['year']+row['month']+row['day']}_HILL.tif"
csv_path = f"temp.csv"
if abs(xres) < res:
    resampleAlg = "bilinear"
elif abs(xres) > res:
    resampleAlg = "cubic"
else:
    resampleAlg = "nearest"
gdal.Warp(tmp_file,raster,format= 'GTiff',resampleAlg=resampleAlg,xRes= res,yRes= res,creationOptions = ['COMPRESS=DEFLATE',"PREDICTOR=2"])
os.system(f"gdal_retile.py -ps {patch_size} {patch_size} -overlap {overlap} -tileIndexField patch -csv {csv_path} -targetDir {out_path} {tmp_file}")
df_patch = pd.read_csv(os.path.join(out_path,csv_path), dtype = str,sep=';',names=["tilename","minx","maxx","miny","maxy"])
df_patch["tile"] = row['tile']
df_patch["area"] = row['area']
df_patch["INSTRUMENT"] = row['INSTRUMENT']
df_patch["SATELLITE"] = row['SATELLITE']
df_patch["year"] = row['year']
df_patch["month"] = row['month']
df_patch["day"] = row['day']
df_patch["band"] = "HILL"
df_patch["type"] = "HILL"
df_patch["row"] = df_patch.tilename.str[-9:-7]
df_patch["col"] = df_patch.tilename.str[-6:-4]
df_patch["path"] = out_path+"/"+df_patch["tilename"]
list_df.append(df_patch)
os.system(f"rm {os.path.join(out_path,csv_path)}")
      
    
#MASKS
raster= gdal.Open(row['MASK_PATH']) 
ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
tmp_file = tmp_dir+f"/{row['tile']}_{row['year']+row['month']+row['day']}_MASK.tif"
csv_path = f"temp.csv"
if abs(xres) < res:
    resampleAlg = "bilinear"
elif abs(xres) > res:
    resampleAlg = "cubic"
else:
    resampleAlg = "nearest"
gdal.Warp(tmp_file,raster,format= 'GTiff',resampleAlg=resampleAlg,xRes= res,yRes= res,creationOptions = ['COMPRESS=DEFLATE']) 
os.system(f"gdal_retile.py -ps {patch_size} {patch_size} -overlap {overlap} -tileIndexField patch -csv {csv_path} -targetDir {out_path} {tmp_file}")
df_patch = pd.read_csv(os.path.join(out_path,csv_path), dtype = str,sep=';',names=["tilename","minx","maxx","miny","maxy"])
df_patch["tile"] = row['tile']
df_patch["area"] = row['area']
df_patch["INSTRUMENT"] = row['INSTRUMENT']
df_patch["SATELLITE"] = row['SATELLITE']
df_patch["year"] = row['year']
df_patch["month"] = row['month']
df_patch["day"] = row['day']
df_patch["band"] = "MASK"
df_patch["type"] = "MASK"
df_patch["row"] = df_patch.tilename.str[-9:-7]
df_patch["col"] = df_patch.tilename.str[-6:-4]
df_patch["path"] = out_path+"/"+df_patch["tilename"]
list_df.append(df_patch)
os.system(f"rm {os.path.join(out_path,csv_path)}")
    
    

#LABELS
raster= gdal.Warp("",glob.glob(os.path.join(row['L2B_PATH'],'**',f"*_SNW_R2.tif"),recursive=True)[0],format= 'MEM') 

array = BandReadAsArray(raster.GetRasterBand(1))
array[(array == 100)] = 1
array[(array == 205)] = 2
array[(array > 205)] = 3

raster.GetRasterBand(1).WriteArray(array)

ulx, xres, xskew, uly, yskew, yres  = raster.GetGeoTransform()
tmp_file = tmp_dir+f"/{row['tile']}_{row['year']+row['month']+row['day']}_L2B.tif"
csv_path = f"temp.csv"
if abs(xres) != res:
    gdal.Warp(tmp_file,raster,format= 'GTiff',resampleAlg="nearest",xRes= res,yRes= res,creationOptions = ['COMPRESS=DEFLATE'])
else:
    gdal.Warp(tmp_file,raster,format="GTiff",creationOptions = ['COMPRESS=DEFLATE']) 
os.system(f"gdal_retile.py -ps {patch_size} {patch_size} -overlap {overlap} -tileIndexField patch -csv {csv_path} -targetDir {out_path} {tmp_file}")
df_patch = pd.read_csv(os.path.join(out_path,csv_path), dtype = str,sep=';',names=["tilename","minx","maxx","miny","maxy"])
df_patch["tile"] = row['tile']
df_patch["area"] = row['area']
df_patch["INSTRUMENT"] = row['INSTRUMENT']
df_patch["SATELLITE"] = row['SATELLITE']
df_patch["year"] = row['year']
df_patch["month"] = row['month']
df_patch["day"] = row['day']
df_patch["band"] = "LABEL"
df_patch["type"] = "LABEL"
df_patch["row"] = df_patch.tilename.str[-9:-7]
df_patch["col"] = df_patch.tilename.str[-6:-4]
df_patch["path"] = out_path+"/"+df_patch["tilename"]
list_df.append(df_patch)
os.system(f"rm {os.path.join(out_path,csv_path)}")
    


        
pd.concat(list_df).to_csv(os.path.join(output_dir,"TMP",f"PATCHES_{index}.csv"), index=False) 
