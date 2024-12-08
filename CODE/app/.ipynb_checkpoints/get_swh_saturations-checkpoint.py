import os
from assumerole import assumerole
import zipfile
import glob
import json
import s3fs
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

#PARSE INPUT ARGUMENTS
print("PARSE INPUT ARGUMENTS")
arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-c', action='store', default="", nargs='?', const='', dest='config_file')
arg_parser.add_argument('-t', action='store', default="", nargs='?', const='', dest='tmp_dir')
config_file = arg_parser.parse_args().config_file
tmp_dir = arg_parser.parse_args().tmp_dir



#ACCESS CONFIGURATION FILE
print("ACCESS CONFIGURATION FILE")
with open(config_file, "r") as f:
    config = json.load(f) 
start_date = datetime.strptime(config["start_date"],'%Y%m%d')
end_date = datetime.strptime(config["end_date"],'%Y%m%d')
bbox = config["bbox"]
work_dir = config["work_dir"]

#ACCESS DATALAKE
print("ACCESS DATALAKE")
ENDPOINT_URL="https://s3.datalake.cnes.fr"
print("get credentials")
credentials = assumerole.getCredentials("arn:aws:iam::732885638740:role/public-read-only-OT", Duration=7200)

print("get s3filesystem")
s3 = s3fs.S3FileSystem(
      client_kwargs={
                      'aws_access_key_id': credentials['AWS_ACCESS_KEY_ID'],
                      'aws_secret_access_key': credentials['AWS_SECRET_ACCESS_KEY'],
                      'aws_session_token': credentials['AWS_SESSION_TOKEN'],
         'endpoint_url': 'https://s3.datalake.cnes.fr'
      }
    
   )

#INITIATE DATAFRAME
print("INITIATE DATAFRAME")
d= {'XS1_SAT':[],'XS1_MIN':[],'XS2_SAT':[],'XS2_MIN':[],'XS3_SAT':[],'XS3_MIN':[],'SWIR_SAT':[],'SWIR_MIN':[],'SATELLITE':[],'K':[],'J':[],'YEAR':[],'MONTH':[]}


#FILL DATAFRAME WITH SWH METADATA FROM DATALAKE
print("FILL DATAFRAME WITH SWH METADATA FROM DATALAKE")
day_count = (end_date - start_date).days + 1
for single_date in (start_date + timedelta(n) for n in range(day_count)): # for each date inside the period
    for K in range(bbox[0],bbox[2]+1):  # for swh products at path K
        for J in range(bbox[1],bbox[3]+1):  # for swh products at row J
            swh_list = s3.glob(f"muscate/SPOTWORLDHERITAGE/{single_date.strftime('%Y')}/{single_date.strftime('%m')}/{single_date.strftime('%d')}/*_{single_date.strftime('%Y%m%d')}-*_L1C_{K:03}-{J:03}-*/*.zip")  # look for swh products with s3fs
            

            for swh in swh_list:
                s3.get(swh, tmp_dir, recursive=True)  #download swh zip in the temporary directory
                #EXTRACT METADATA INFORMATION
                zip_swh= zipfile.ZipFile(os.path.join(tmp_dir,os.path.basename(swh)))
                for ziped_file in zip_swh.namelist():
                    if "MTD_ALL.xml" in ziped_file and ".tif.aux.xml" not in ziped_file :
                        f = zip_swh.read(ziped_file)
                        tree= ET.fromstring(f)
                        sat_name = ((tree.findall(".//Product_Characteristics/PLATFORM"))[0]).text
                        ref_max_XS1 = float(((tree.findall(".//Band_Index_List[@band_id='XS1']/QUALITY_INDEX[@name='ReflectanceMax']"))[0]).text)
                        ref_min_XS1 = float(((tree.findall(".//Band_Index_List[@band_id='XS1']/QUALITY_INDEX[@name='ReflectanceMin']"))[0]).text)
                        ref_max_XS2 = float(((tree.findall(".//Band_Index_List[@band_id='XS2']/QUALITY_INDEX[@name='ReflectanceMax']"))[0]).text)
                        ref_min_XS2 = float(((tree.findall(".//Band_Index_List[@band_id='XS2']/QUALITY_INDEX[@name='ReflectanceMin']"))[0]).text)
                        ref_max_XS3 = float(((tree.findall(".//Band_Index_List[@band_id='XS3']/QUALITY_INDEX[@name='ReflectanceMax']"))[0]).text)
                        ref_min_XS3 = float(((tree.findall(".//Band_Index_List[@band_id='XS3']/QUALITY_INDEX[@name='ReflectanceMin']"))[0]).text)
                        if "SPOT4" in sat_name or "SPOT5" in sat_name:
                            ref_max_SWIR = float(((tree.findall(".//Band_Index_List[@band_id='SWIR']/QUALITY_INDEX[@name='ReflectanceMax']"))[0]).text)
                            ref_min_SWIR = float(((tree.findall(".//Band_Index_List[@band_id='SWIR']/QUALITY_INDEX[@name='ReflectanceMin']"))[0]).text)
                        else: 
                            ref_max_SWIR = np.nan
                            ref_min_SWIR = np.nan
                        d['XS1_SAT'].append(ref_max_XS1) 
                        d['XS2_SAT'].append(ref_max_XS2)
                        d['XS3_SAT'].append(ref_max_XS3)
                        d['SWIR_SAT'].append(ref_max_SWIR)
                        d['XS1_MIN'].append(ref_min_XS1)
                        d['XS2_MIN'].append(ref_min_XS2)
                        d['XS3_MIN'].append(ref_min_XS3)
                        d['SWIR_MIN'].append(ref_min_SWIR)
                        d['MONTH'].append(single_date.month)
                        d['YEAR'].append(single_date.year)
                        d['K'].append(K)
                        d['J'].append(J)
                        d['SATELLITE'].append(sat_name)
                print(os.path.basename(swh))
                os.system(f"rm {os.path.join(tmp_dir,os.path.basename(swh))}") #delete swh zip product from temporary directory
            
#SAVE DATAFRAME
print("SAVE DATAFRAME")
df = pd.DataFrame(data=d)
common_name = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{bbox[0]}-{bbox[2]}_{bbox[1]}-{bbox[3]}"
df.to_csv(os.path.join(work_dir,f"SPOT_SAT_{common_name}.csv"), index=False)           
            
