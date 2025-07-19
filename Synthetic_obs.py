import logging
import warnings
# Removed redundant import
import os
from os.path import join
import shutil

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt 

import json
import time


os.chdir(os.getenv("EWC_RUNDIR", "/home/pwiersma/scratch/Scripts/Github/ewc"))
from SwissStations import *
from SnowClass import *
from RunWflow_Julia import *

from pathlib import Path
import numpy as np
from IPython.display import clear_output

from scipy.ndimage import gaussian_filter

def additive_gaussian_noise(Q,scale):
    noise = np.random.normal(size = len(Q),scale = scale,loc = 0)
    noise_corr = gaussian_filter(noise,1)
    return Q + noise_corr

def multiplicative_gaussian_noise(Q,scale):
    noise = np.random.normal(size = len(Q),scale = scale,loc = 1)
    noise_corr = gaussian_filter(noise,1)
    return Q * noise_corr

class Synthetic_obs:
    def __init__(self, config, RUN_NAME):
        self.cfg = config
        self.RUN_NAME = RUN_NAME
        for key, value in config.items():
            setattr(self, key, value)
        for key,value in config['OBS_ERROR'].items():
            setattr(self,key,value)
        self.START_YEAR = config['START_YEAR']-1 #spinup year
        self.END_YEAR = config['END_YEAR']

        #Create sfcf for each year based on sfcf_start
        # sfcf_start = self.trueparams['sfcf_start']
        # direction = 1
        # for year in range(self.START_YEAR,self.END_YEAR+1):
        #     self.trueparams[f"sfcf_{year}"] = np.round(sfcf_start,4)
        #     if self.trueparams[f"sfcf_{year}"] in  self.param_ranges['sfcf']:
        #         direction *= -1
        #     sfcf_start += self.trueparams['sfcf_change']*direction
        # #remove sfcf_start and sfcf_change from trueparams
        # self.trueparams.pop('sfcf_start')
        # self.trueparams.pop('sfcf_change')

        # print("True params:",self.trueparams)

        # self.generate_synthetic_obs()

    def generate_synthetic_obs(self):
        time0 = time.time()

        #Without OSHD
        if self.SYN_SNOWMODEL == 'Hock':
            params = self.trueparams.copy()
            params['DD'] = 'Hock'
            params['masswasting'] = True
            params['petcf_seasonal'] = True
            params[self.OSHD] = False
            YEARLY_PARAMS = True

        elif self.SYN_SNOWMODEL == 'seasonal':
            params = self.trueparams.copy()
            params['DD'] = 'seasonal'
            params['masswasting'] = True
            params['petcf_seasonal'] = True
            params[self.OSHD] = False
            YEARLY_PARAMS = True
        
        #With OSHD 
        elif self.SYN_SNOWMODEL == 'OSHD':
            params = self.trueparams.copy()
            params['DD'] = 'static'
            params['masswasting'] = False
            params['petcf_seasonal'] = True
            params['sfcf'] = 0
            params[self.OSHD] = True
            YEARLY_PARAMS = False

        # self.RUN_NAME = f"test5"

        model_dics = {}
        model_dics[f"{self.RUN_NAME}"] = params
                
        #
        t0 = time.time()
        wflow = RunWflow_Julia(
            ROOTDIR = self.ROOTDIR,
            PARSETDIR = self.EXPDIR,
            BASIN = self.BASIN,
            RUN_NAME = self.RUN_NAME,
            START_YEAR = self.START_YEAR,
            END_YEAR = self.END_YEAR,
            CONFIG_FILE = "sbm_config_CH_orig.toml",
            RESOLUTION = self.RESOLUTION,
            YEARLY_PARAMS=YEARLY_PARAMS,
            CONTAINER = self.CONTAINER,
            NC_STATES=self.NC_STATES)
        
        Path(self.SYNDIR).mkdir(parents = True,exist_ok=True)        

        forcing_name = f"wflow_MeteoSwiss_{wflow.RESOLUTION}_{self.BASIN}_{self.START_YEAR-1}_{self.END_YEAR}.nc"
        staticmaps_name = f"staticmaps_{wflow.RESOLUTION}_{self.BASIN}_feb2024.nc"
        wflow.generate_MS_forcing_if_needed(forcing_name)
        wflow.check_staticmaps(staticmaps_name)
        wflow.load_model_dics(model_dics)
        wflow.adjust_config()    
        wflow.create_models()
        initialize_time = time.time()-t0
        print(f"Initialization takes {initialize_time} seconds")

        wflow.series_runs(wflow.standard_run, test = False)
        wflow.finalize_runs()
        # wflow.save_state(list(model_dics.keys())[0],parsetname)
        wflow.load_stations([self.BASIN])
        wflow.load_Q()
        wflow.stations_combine()
        time1 = time.time()- time0

        wflow.save_postrun(FOLDER = self.SYNDIR, 
                           vars = self.NC_STATES,
                           only_Q = False)


        print(f"Years altogether takes {time1} seconds ({time1/60}) minutes")

        # Q = wflow.stations[self.BASIN].combined.loc[slice(f"{self.START_YEAR}-10-01",
        #                                                                 f"{self.END_YEAR}-09-30")]
        # Q.plot()

        #Save Q to station format

        
        Q= wflow.stations[self.BASIN].combined[self.RUN_NAME].loc[slice(f"{self.START_YEAR}-10-01",
                                                                        f"{self.END_YEAR}-09-30")] #remove spinup year
        #Add noise to Q
        if self.kind =='multiplicative':
            Q_adjusted = multiplicative_gaussian_noise(Q,self.scale)
            Q[Q<0] = 0
        elif self.kind == 'additive':
            Q_adjusted = additive_gaussian_noise(Q,self.scale)
            Q[Q<0] = 0
        Q_adjusted.to_csv(join(self.SYNDIR,f"Q_{self.EXP_NAME}_{self.scale}.csv"))

        f1,ax1 = plt.subplots(figsize = (10,5))
        Q[100:300].plot(ax = ax1, label = 'Q_orig')
        Q_adjusted[100:300].plot(ax = ax1, label = 'Q_adjusted')
        ax1.set_ylabel('Q [m3/s]')
        ax1.legend()
        ax1.grid()
        ax1.set_title(f"{self.kind} noise - {self.EXP_NAME} - scale = {self.scale}")
        f1.savefig(join(self.SYNDIR,f"Q_{self.EXP_NAME}_{self.scale}.png"))

        #copy SWE output to syntheitc folder
        # output_file =os.path.join(wflow.MODEL_DICS[self.RUN_NAME]['outfolder'], wflow.MODEL_DICS[self.RUN_NAME]['config']['output']['path'])
        # shutil.copy(output_file,join(self.SYNDIR,f"SWE_{self.EXP_NAME}.nc"))

        # #copy scalars to synthetic folder 
        # scalars_file = os.path.join(wflow.MODEL_DICS[self.RUN_NAME]['outfolder'], wflow.MODEL_DICS[self.RUN_NAME]['config']['csv']['path'])
        # shutil.copy(scalars_file,join(self.SYNDIR,os.path.basename(scalars_file)))

        #Save params to synthetic folder with json dump
        with open(join(self.SYNDIR,f"params_{self.EXP_NAME}.json"), 'w') as fp:
            json.dump(params, fp)
        
        print("All synthetic files written")
            
        

