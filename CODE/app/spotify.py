print("import")
import os
import os.path as op
import sys
import errno
import re
import json
import shutil
import glob
import argparse
import copy
from datetime import datetime, timedelta, date
from osgeo import osr, gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
from scipy.stats import gaussian_kde as GKDE
import cloudpickle
import argparse
import copy
import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt




mtn_tiles={
    "PYR":
    {
        "30TXN":{'EPSG':'32630','MINX':600000,'MINY':4690200,'MAXX':709800,'MAXY':4800000},
        '30TYN':{'EPSG':'32630','MINX':699960,'MINY':4690200,'MAXX':809760,'MAXY':4800000},
        '31TCH':{'EPSG':'32631','MINX':300000,'MINY':4690200,'MAXX':409800,'MAXY':4800000},
        '31TDH':{'EPSG':'32631','MINX':399960,'MINY':4690200,'MAXX':509760,'MAXY':4800000}
    },
    "ALP":
    {
        "31TGJ":{'EPSG':'32631','MINX':699960,'MINY':4790220,'MAXX':809760,'MAXY':4900020},
        '31TGK':{'EPSG':'32631','MINX':699960,'MINY':4890240,'MAXX':809760,'MAXY':5000040},
        '31TGL':{'EPSG':'32631','MINX':699960,'MINY':4990200,'MAXX':809760,'MAXY':5100000},
        '31TGM':{'EPSG':'32631','MINX':699960,'MINY':5090220,'MAXX':809760,'MAXY':5200020},
        "32TLP":{'EPSG':'32632','MINX':300000,'MINY':4790220,'MAXX':409800,'MAXY':4900020},
        '32TLQ':{'EPSG':'32632','MINX':300000,'MINY':4890240,'MAXX':409800,'MAXY':5000040},
        '32TLR':{'EPSG':'32632','MINX':300000,'MINY':4990200,'MAXX':409800,'MAXY':5100000},
        '32TLS':{'EPSG':'32632','MINX':300000,'MINY':5090220,'MAXX':409800,'MAXY':5200020}
    }
}





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
work_dir = config["work_dir"]
target_sat = config["target_sat"]
target_ins = config["target_ins"]
add_sat = config["add_sat"]
index = str(config["index"])
redraws_nb = config["draws"]
library_path = config["library"]


#GET DATAFRAME
print("GET DATAFRAME")
df = (pd.read_csv(input_csv, dtype = str)).query(f"i == '{index}'")
input_sat = df['SATELLITE'].iloc[0]
input_ins = df['INSTRUMENT'].iloc[0]
df["PSEUDO_SATELLITE"] = target_sat
df["PSEUDO_INSTRUMENT"] = target_ins
#GET LIBRARY
library = json.load(open(library_path))



#get drivers
tiff_out = gdal.GetDriverByName('GTiff')





print("adjust plot sizes")
#PLOT SIZES#######################
SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



print("search overlaps")
#SEARCH OVERLAPS ####################################"
dict_overlaps = {}
for target_band in sorted(library["SWH"][target_sat]["INSTRUMENTS"][target_ins]["BANDS"]):
    target_center = float(library["SWH"][target_sat]["INSTRUMENTS"][target_ins]["BANDS"][target_band]["BAND CENTER"])
    target_width = float(library["SWH"][target_sat]["INSTRUMENTS"][target_ins]["BANDS"][target_band]["BAND WIDTH"])
    target_min = target_center - target_width/2
    target_max = target_center + target_width/2
    #total_overlap = 0
    dict_overlaps[target_band] = {}
    dict_overlaps[target_band]["band_overlaps"] = {}
    for input_band in sorted(library[input_sat][input_ins]["BANDS"]):
        input_center = float(library[input_sat][input_ins]["BANDS"][input_band]["BAND CENTER"])
        input_width = float(library[input_sat][input_ins]["BANDS"][input_band]["BAND WIDTH"])
        input_min = input_center - input_width/2
        input_max = input_center + input_width/2
        overlap_min = max(input_min,target_min)
        overlap_max = min(input_max,target_max)
        if overlap_min < overlap_max:
            overlap = overlap_max - overlap_min
            dict_overlaps[target_band]["band_overlaps"][input_band] = overlap
            #total_overlap = total_overlap + overlap
    #dict_overlaps[target_band]["total_overlap"] = total_overlap



print("conversion")
#CONVERSION################################################"
#for each crop############
list_rows=[]


for area in df["area"].unique():
    #GET GKDE
    sat_file = open(os.path.join(work_dir,"DATA","SWH_SATURATIONS",area,"GKDE",f"{target_sat}_SAT_GKDE.pkl"),'rb')
    sat_gkde = cloudpickle.load(sat_file)
    min_file = open(os.path.join(work_dir,"DATA","SWH_SATURATIONS",area,"GKDE",f"{target_sat}_MIN_GKDE.pkl"),'rb')
    min_gkde = cloudpickle.load(min_file)
    
    for _ , row in df.query(f"area == '{area}'").iterrows():
        itile = row["row"]
        jtile = row["col"]

        sat_refs = sat_gkde.resample(size=redraws_nb)
        min_refs = min_gkde.resample(size=redraws_nb)


        #for each draw##################
        for draw_id in range(redraws_nb):
            drow = row.copy(deep=True)
            drow['draw'] = draw_id+1




            list_sats = []
            #for each target band to simulate#######
            for target_band in sorted(dict_overlaps):   
                target_ID = library["SWH"][target_sat]["INSTRUMENTS"][target_ins]["BANDS"][target_band]["ID"]
                target_name = library["SWH"][target_sat]["INSTRUMENTS"][target_ins]["BANDS"][target_band]["NAME"]
                total_band_array = None
                input_band_raster = None
                total_overlap = 0
                target_max_ref_saturation = 0
                target_min_ref_saturation = 0
                sat_index = 0
                if target_ID == "XS1":
                    target_max_ref_saturation = int(sat_refs[0][draw_id]*1000)
                    target_min_ref_saturation = int(min_refs[0][draw_id]*1000)
                    sat_index = 1
                elif target_ID == "XS2":
                    target_max_ref_saturation = int(sat_refs[1][draw_id]*1000)
                    target_min_ref_saturation = int(min_refs[1][draw_id]*1000)
                    sat_index = 2
                elif target_ID == "XS3":
                    target_max_ref_saturation = int(sat_refs[2][draw_id]*1000)
                    target_min_ref_saturation = int(min_refs[2][draw_id]*1000)
                    sat_index = 4
                elif target_ID == "SWIR":
                    target_max_ref_saturation = int(sat_refs[3][draw_id]*1000)
                    target_min_ref_saturation = int(min_refs[3][draw_id]*1000)
                    sat_index = 8
                else:
                    continue

                print(itile,jtile,draw_id,target_band)
                #calculate equivalent target band from the input overlapping bands#########
                for input_band in sorted(dict_overlaps[target_band]["band_overlaps"]):
                    input_ID = library[input_sat][input_ins]["BANDS"][input_band]["ID"]
                    input_name = library[input_sat][input_ins]["BANDS"][input_band]["NAME"]
                    overlap = dict_overlaps[target_band]["band_overlaps"][input_band]
                    if input_ID not in df.columns:
                        print("WARNING: could not find a raster for the band {} for the crop {} {} for target {}".format(input_ID,itile,jtile,target_band)) 
                    else:
                        input_band_raster = gdal.Translate('',row[input_ID],format='MEM')
                        input_band_array = BandReadAsArray(input_band_raster.GetRasterBand(1))
                        weighted_band_array = overlap * input_band_array
                        if  total_band_array is None:
                            total_band_array = weighted_band_array
                        else:
                            total_band_array = total_band_array + weighted_band_array
                        total_overlap = total_overlap + overlap
                        print("found raster for the band {} for the crop {} {} for target {}".format(input_ID,itile,jtile,target_band))

                #if all overlaping bands are available#################"
                if total_overlap > 0:
                    print("S2 TO SPOT for the crop {} {} for target {}".format(itile,jtile,target_band))


                    #saturate equivalent target band ###############################"
                    out_path=os.path.join(work_dir,"TRAINING","S2","PATCHES",row['tile'],row['year'],row['month'],row['day'])
                    os.system(f"mkdir -p {out_path}")
                    #total_overlap = dict_overlaps[target_band]["total_overlap"] 
                    average_band_array = total_band_array/total_overlap
                    target_band_raster = input_band_raster
                    target_band_raster.GetRasterBand(1).WriteArray(average_band_array)
                    common_name = "{}_{}_TO_{}_{}_".format(input_sat,input_ins,target_sat,target_ins)  + "{}_{}_{}_{}_{}_DRAW_{}.tif".format(target_ID,row['tile'],row['year']+row['month']+row['day'],itile,jtile,draw_id+1)
                    sat_name = "{}_{}_TO_{}_{}_".format(input_sat,input_ins,target_sat,target_ins)  + "SAT{}_{}_{}_{}_{}_DRAW_{}.tif".format(target_band,row['tile'],row['year']+row['month']+row['day'],itile,jtile,draw_id+1)
                    formated_band_path = op.join(out_path,"FORMATED_"+common_name)

                    DN_path = op.join(out_path,"DN_"+common_name)
                    os.system("rm -f "+formated_band_path+" "+DN_path)
                    #print("original to DN")
                    target_DN = gdal.Translate(DN_path, target_band_raster, format='GTiff', outputType=gdal.GDT_Byte, scaleParams=[[target_min_ref_saturation*10,target_max_ref_saturation*10,0,255]])
                    #print("DN to target")
                    formated_band = gdal.Translate(formated_band_path, target_DN, format='GTiff', outputType=gdal.GDT_UInt16,  scaleParams=[[0,255,target_min_ref_saturation,target_max_ref_saturation]])
                    drow[target_ID]= formated_band_path

                    if add_sat == 1:
                        saturation_band_path = op.join(out_path,"SATURATION_"+sat_name)
                        DN_array = copy.copy(BandReadAsArray(target_DN.GetRasterBand(1)))
                        DN_array[DN_array != 255] = 0
                        DN_array[DN_array == 255] = 1
                        target_DN.GetRasterBand(1).WriteArray(DN_array)
                        gdal.Translate(saturation_band_path, target_DN, format='GTiff')
                        DN_array[DN_array == 1] = sat_index

                        #saturation_array = np.where(DN_array == 255 , sat_index, 0)
                        list_sats.append(DN_array)
                        print(DN_array.shape, DN_array.max())


                    os.system("rm "+DN_path)



                else:
                    print("no available input bands for the crop {} {} for target {}".format(itile,jtile,target_band))
                    drow[target_ID]= np.nan


            #create saturation raster
            if add_sat == 1:
                saturation_multi_band_path = op.join(out_path,"SATMULTI_"+"{}_{}_TO_{}_{}_".format(input_sat,input_ins,target_sat,target_ins)+ "{}_{}_{}_{}_DRAW_{}.tif".format(row['tile'],row['year']+row['month']+row['day'],itile,jtile,draw_id+1))

                saturations_array = np.sum(np.dstack(list_sats),axis=2)
                #saturations_array = sum(list_sats)
                print(saturations_array.shape)

                target_DN.GetRasterBand(1).WriteArray(saturations_array)
                gdal.Translate(saturation_multi_band_path, target_DN, format='GTiff')

            list_rows.append(drow)
        
pd.DataFrame(list_rows).to_csv(os.path.join(work_dir,"TMP",f"PATCHES_{index}_{target_sat}_{target_ins}.csv"), index=False) 
