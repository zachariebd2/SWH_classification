import os
import glob
import json
import argparse
from scipy.stats import gaussian_kde as GKDE
import numpy as np
import cloudpickle
import pandas as pd



#PARSE INPUT ARGUMENTS
print("PARSE INPUT ARGUMENTS")
arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-c', action='store', default="", nargs='?', const='', dest='config_file')
config_file = arg_parser.parse_args().config_file




#ACCESS CONFIGURATION FILE
print("ACCESS CONFIGURATION FILE")
with open(config_file, "r") as f:
    config = json.load(f) 
bw = config["bw"]
work_dir = config["work_dir"]



#GET DATAFRAME
print("GET DATAFRAME")
df_csvs = pd.DataFrame()
for csv_file in glob.glob(os.path.join(work_dir,'*.csv')):
    df_csvs = df_csvs.append(pd.read_csv(csv_file), ignore_index=True)


#MAKE GKDE DISTRIBUTION 
gkde_dir = os.path.join(work_dir,"GKDE")
os.system(f"mkdir -p {gkde_dir}")
print("MAKE GKDE DISTRIBUTION ")
for satellite in ["SPOT1","SPOT2","SPOT3"]:
    df1 = df_csvs.query("SATELLITE == '"+satellite+"'")
    if len(df1)>=2:
        XS1 = df1['XS1_SAT']
        XS2 = df1['XS2_SAT']
        XS3 = df1['XS3_SAT']
        sat_XS123 = np.vstack([XS1, XS2,XS3])
        kernel_sat_XS123 = GKDE(sat_XS123,bw_method=bw) 
        GKDE_SAT_path = os.path.join(gkde_dir,f"{satellite}_SAT_GKDE.pkl")
        os.system("rm "+GKDE_SAT_path)
        print(f"write {GKDE_SAT_path}")
        cloudpickle.dump(kernel_sat_XS123, open(GKDE_SAT_path, 'wb'))
        MIN1 = df1['XS1_MIN']
        MIN2 = df1['XS2_MIN']
        MIN3 = df1['XS3_MIN']
        min_XS123 = np.vstack([MIN1, MIN2,MIN3])
        kernel_min_XS123 = GKDE(min_XS123,bw_method=bw)
        GKDE_MIN_path = os.path.join(gkde_dir,f"{satellite}_MIN_GKDE.pkl")
        os.system("rm "+GKDE_MIN_path)
        print(f"write {GKDE_MIN_path}")
        cloudpickle.dump(kernel_min_XS123, open(GKDE_MIN_path, 'wb'))

for satellite in ["SPOT4","SPOT5"]:
    df1 = df_csvs.query("SATELLITE == '"+satellite+"'")
    if len(df1)>=2:
        XS1 = df1['XS1_SAT']
        XS2 = df1['XS2_SAT']
        XS3 = df1['XS3_SAT']
        SWIR = df1['SWIR_SAT']
        sat_XS1234 = np.vstack([XS1, XS2,XS3,SWIR])
        kernel_sat_XS1234 = GKDE(sat_XS1234,bw_method=bw)
        GKDE_SAT_path = os.path.join(gkde_dir,f"{satellite}_SAT_GKDE.pkl")
        os.system("rm "+GKDE_SAT_path)
        print(f"write {GKDE_SAT_path}")
        cloudpickle.dump(kernel_sat_XS1234, open(GKDE_SAT_path, 'wb'))
        MIN1 = df1['XS1_MIN']
        MIN2 = df1['XS2_MIN']
        MIN3 = df1['XS3_MIN']
        SWIR = df1['SWIR_MIN']
        min_XS1234 = np.vstack([MIN1, MIN2,MIN3,SWIR])
        kernel_min_XS1234 = GKDE(min_XS1234,bw_method=0.1)
        GKDE_MIN_path = os.path.join(gkde_dir,f"{satellite}_MIN_GKDE.pkl")
        os.system("rm "+GKDE_MIN_path)
        print(f"write {GKDE_MIN_path}")
        cloudpickle.dump(kernel_min_XS1234, open(GKDE_MIN_path, 'wb'))

