import pandas as pd
import os
import glob
import xml.etree.ElementTree as ET
import argparse

#PARSE INPUT ARGUMENTS
print("PARSE INPUT ARGUMENTS")
arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-c', action='store', default="", nargs='?', const='', dest='csv_file')
csv_file = arg_parser.parse_args().csv_file

df = pd.read_csv(csv_file, dtype = str)
for index, row in df.iterrows():
    MTD=glob.glob(os.path.join(row['L1C_PATH'],"**","MTD_TL.xml"),recursive=True)[0]
    tree = ET.parse(MTD)
    root = tree.getroot()
    altitude = 90 - float((root.findall(".//Mean_Sun_Angle/ZENITH_ANGLE[@unit='deg']")[0]).text)
    azimuth =  float((root.findall(".//Mean_Sun_Angle/AZIMUTH_ANGLE[@unit='deg']")[0]).text)
    os.system(f"mkdir -p {os.path.dirname(row['HILL_PATH'])}")
    os.system(f"gdaldem hillshade {row['DEM_PATH']} {row['HILL_PATH']} -alg Horn -az {azimuth} -alt {altitude} -compute_edges")