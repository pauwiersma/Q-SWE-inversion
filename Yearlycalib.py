



import os
import sys
import json
import pandas as pd
from os.path import join
from spotpy_calib import YearlyCalib 

# basin = sys.argv[1]
# year = int(sys.argv[2])
# cfg = json.load(open(sys.argv[3]))
# cfg['BASIN'] = basin

EWC_ROOTDIR = os.getenv('EWC_ROOTDIR', default = '/home/pwiersma/scratch/Data/ewatercycle/')
EXP_ID = "Syn_912"
BASIN = "Dischma"
TEST = True
CFG_DIR = f"{EWC_ROOTDIR}/experiments/config_files/{EXP_ID}_config.json"
cfg = json.load(open(CFG_DIR))
JOB_ID = 0
year = 2018

FOLDER = join(cfg['OUTDIR'],f"Soilcalib")
FILE = join(FOLDER, f"{cfg['BASIN']}_ksoil{cfg['SOIL_K']}_posterior.csv")
cluster_parsets = pd.read_csv(FILE,header=0, index_col=0)
cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')


for m,pars  in cluster_parsets_dic.items():
    print(m,pars)
    # cluster_parset = cluster_parsets_dic[int(m)]

    YearlyCalib(cfg, year, m, pars)