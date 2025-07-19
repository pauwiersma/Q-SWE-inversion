
import logging
import warnings
import numpy as np
import pandas as pd
import glob
import os
from os.path import join
import rasterio as rio
import tomlkit
import shutil

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
import xarray as xr
import rioxarray as rioxr
# from scipy import interpolate
import matplotlib.pyplot as plt 
import time
from datetime import date
from datetime import datetime
import HydroErr as he
import json
import time

RUNDIR = os.getenv('EWC_RUNDIR', default = '/home/pwiersma/scratch/Scripts/Github/ewc/')
os.chdir(RUNDIR)

from SwissStations import *
from SnowClass import *
from RunWflow_Julia import *

from wflow_spot_setup import *

from pathlib import Path
import numpy as np
from IPython.display import clear_output
import spotpy.analyser as spa

class spotpy_analysis(object):
    """ Should contain functions for
    - Making clusters and saving them
    - Plotting
        Normalized parameter plots for each parameter (prior + posterior)
        OF development + cutoff threshold
        Hydrographs for each year from prior and posterior clusters
        Parameter interaction plot (with prior and posterior colours)
        Parameter development
        Relative bias 

        """
    # def __init__(self,
    #              config,
    #              RUN_NAME,
    #              calibration_purpose,
    #              single_year = None,
    #              member = None,):
    def __init__(self, 
                 config, 
                 RUN_NAME, 
                 calibration_purpose, 
                 single_year=None, 
                 member=None):
        self.cfg = config
        self.RUN_NAME = RUN_NAME
        self.calibration_purpose = calibration_purpose
        self.single_year = single_year
        self.member = member

    def run(self):
        self._set_attributes_from_config()
        self._initialize_paths_and_files()
        self._load_results()
        self._process_posterior()
        self._calculate_clusters()
        self._plot_clusters()
        self._plot_metric_evolution()
        self._plot_parameter_trace_and_interaction()

    def _set_attributes_from_config(self):
        for key, value in self.cfg.items():
            setattr(self, key, value)
        for key, value in self.cfg['OBS_ERROR'].items():
            setattr(self, key, value)

    def _initialize_paths_and_files(self):
        if self.calibration_purpose == 'Soilcalib':
            self.plot_folder = join(self.SOILDIR, f"{self.BASIN}_Plots")
            Path(self.plot_folder).mkdir(parents=True, exist_ok=True)
            self.results_file = join(self.SOILDIR, f"{self.RUN_NAME}")
            self.k = self.cfg['SOIL_K']
        elif self.calibration_purpose == 'Yearlycalib':
            self.plot_folder = join(self.YEARLYDIR, f"{self.single_year}", f"{self.member}", f"{self.BASIN}_Plots")
            Path(self.plot_folder).mkdir(parents=True, exist_ok=True)
            self.results_file = join(self.YEARLYDIR, f"{self.single_year}", f"{self.member}", f"{self.RUN_NAME}")
            self.k = self.cfg['YEARLY_K']

    def _load_results(self):
        self.setup = spot_setup(self.cfg, self.RUN_NAME, calibration_purpose=self.calibration_purpose)
        self.results = spa.load_csv_results(self.results_file)
        self.params = spa.get_parameters(self.results)
        self.parnames = spa.get_parameternames(self.results)
        self.evaluation = self.setup.evaluation()

    def _load_Q(self):
        sims = pd.DataFrame(spa.get_modelruns(self.results)).transpose()
        self.Q = sims

    def _process_posterior(self):
        self.posterior = spa.get_posterior(self.results, percentage=self.POSTERIOR_PCT * 100)
        self.posterior_df = pd.DataFrame(spa.get_parameters(self.posterior))
        self.posterior_df[self.OF] = self.posterior['like1']
        self.posterior_df = self.posterior_df.sort_values(by=self.OF, ascending=False).reset_index(drop=True)

        if self.OF_range is not None:
            self._adjust_posterior_range()

        self.posterior_df.columns = self.posterior_df.columns.str.replace('par', '')
        self._save_posterior_df()

    def _adjust_posterior_range(self):
        l1 = 0
        additional_range = 0.00
        while l1 < 5 * self.k:
            l0 = len(self.posterior_df)
            posterior_k = self.posterior_df[self.posterior_df[self.OF] > self.posterior_df[self.OF].max() - self.OF_range - additional_range]
            l1 = len(posterior_k)
            print("ensemble_size", l1)
            print("additional range", additional_range)
            print(self.single_year, self.member)
            additional_range += 0.005
            if additional_range > 10:
                self.OF_range = None
                break
        self.posterior = spa.get_posterior(self.results, percentage=(l1 / len(self.results)) * 100)
        print("additional_range", additional_range)
        self.additional_range = additional_range
        self.posterior_df = posterior_k.copy()

    def _save_posterior_df(self):
        if self.calibration_purpose == 'Soilcalib':
            self.posterior_df.to_csv(join(self.OUTDIR, f"{self.calibration_purpose}", f"{self.BASIN}_posterior_parsets.csv"))
        elif self.calibration_purpose == 'Yearlycalib':
            self.posterior_df.to_csv(join(self.OUTDIR, f"{self.calibration_purpose}", f"{self.single_year}", f"{self.member}", f"{self.BASIN}_ms{self.member}_posterior_parsets.csv"))

    def _calculate_clusters(self):
        self.calc_clusters(self.posterior, 'posterior')
        if self.N > 50:
            self.calc_clusters(self.results[:int(self.PCT_FIRST_RUNS * self.N)], 'prior')
        else:
            self.calc_clusters(self.results, 'prior')

    def _plot_clusters(self):
        self.plot_clusters('prior')
        self.plot_clusters('posterior')

    def _plot_metric_evolution(self):
        f1, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(self.results['like1'])
        ax1.set_ylabel(self.OF)
        ax1.set_xlabel('Run number')
        ax1.set_title(f"Best OF =  {self.results['like1'].max()}")
        ax1.grid()
        if self.OF_range is not None:
            ax1.axhline(self.results['like1'].max() - self.OF_range - self.additional_range, color='black', linestyle='dashed', label='Posterior ensemble threshold')
            ax1.legend()
        plt.savefig(os.path.join(self.plot_folder, 'Metric_evolution.png'))

    def _plot_parameter_trace_and_interaction(self):
        spa.plot_parametertrace(self.results, figsize=(10, 10), labelpad=30, fig_name=os.path.join(self.plot_folder, 'Parametertrace.png'))
        spa.plot_parameterInteraction(self.results, fig_name=os.path.join(self.plot_folder, 'Parameter_interaction.png'))
       
    def calc_clusters(self, parsets,dist): 
        from sklearn import cluster
        df = pd.DataFrame(spotpy.analyser.get_parameters(parsets))
        df[self.OF] = parsets['like1']
        df = df.sort_values(by = self.OF,ascending=False).reset_index(drop = True)
        print(df)
        df.columns = df.columns.str.replace('par','')
        df.pop(self.OF)
        
        pars_scaled = pd.DataFrame()
        for col in df.columns:
            minimum,maximum = self.param_ranges[col]
            pars_scaled[col] = (df[col] - minimum)/(maximum-minimum)

        kmeans = cluster.KMeans(n_clusters = self.k, random_state = 1)
        kmeans = kmeans.fit(pars_scaled)

        cluster_assignments = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        closest_param_sets = np.empty((self.k, pars_scaled.shape[1]))

        # Loop through each cluster
        for cluster_id in range(self.k):
            # Find the indices of parameter sets in the current cluster
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]

            # Calculate distances from all parameter sets in the cluster to the centroid
            distances = np.linalg.norm(pars_scaled.values[cluster_indices] - cluster_centers[cluster_id], axis=1)

            # Find the index of the parameter set with the minimum distance
            closest_index = cluster_indices[np.argmin(distances)]

            # Store the closest parameter set
            closest_param_sets[cluster_id] = pars_scaled.values[closest_index]

        cluster_parsets_normed = pd.DataFrame(data = closest_param_sets, columns = pars_scaled.columns)#.to_dict(orient = 'index')
        cluster_parsets = pd.DataFrame()
        for col in cluster_parsets_normed.columns:
            minimum,maximum = self.param_ranges[col]
            cluster_parsets[col] = (cluster_parsets_normed[col] * (maximum -minimum)) + minimum
        # cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')
        # cluster_parsets.to_csv(join(PARSETDIR,"spotpy_outputs",f"{spotpy_name}_{basin}_{spotpy_suffix}_{self.k}_clusters.csv"))
        if self.calibration_purpose == 'Soilcalib':
            cluster_parsets.to_csv(join(self.OUTDIR,f"{self.calibration_purpose}",
                                        f"{self.BASIN}_ksoil{self.k}_{dist}.csv"))
        elif self.calibration_purpose == 'Yearlycalib':
            cluster_parsets.to_csv(join(self.OUTDIR,f"{self.calibration_purpose}",
                                        f"{self.single_year}",
                                        f"{self.member}",
                                        f"{self.BASIN}_ms{self.member}_kyearly{self.k}_{dist}.csv"))
        setattr(self, f"clusters_{dist}", cluster_parsets)
        setattr(self, f"clusters_normed_{dist}", cluster_parsets_normed)
        setattr(self, f"pars_scaled_{dist}", pars_scaled)
        setattr(self, f"pars_df_{dist}", df)


    def plot_clusters(self,dist):
        N_axes = len(self.parnames)
        pars_scaled =  getattr(self, f"pars_scaled_{dist}")
        clusters_normed = getattr(self, f"clusters_normed_{dist}")
        ## Ploting
        f1,axes = plt.subplots(3,int(np.ceil(N_axes/3)),figsize = (15,9),sharey = True)
        plt.subplots_adjust(hspace = 0.3)
        axes = axes.flatten()
        # plt.suptitle(basin + ' normalized posterior parameters + ' +str(k)+' clusters', y = 1.1)
        for i,col in enumerate(self.parnames):
            ax = axes[i]
            # s = pars_scaled[col].sort_values().reset_index()
            # ax.plot(s)
            pars_scaled[col].plot(ax = ax)
            ax.set_title(col)
            ax.set_ylim([0,1])
            
            twin = plt.twinx(ax)
            Npars = len(pars_scaled)
            x = np.arange(0,Npars,Npars/self.k)+ (Npars/self.k)/2
            # twin.scatter(x = x,y = cluster_centers[:,i], c= x, cmap = 'turbo')
            twin.scatter(x = x,y = clusters_normed[col], c= x, cmap = 'turbo')

            twin.set_ylim([0,1])
            twin.set_yticks([])
            if i==0:
                ax.set_ylabel('Normalized parameter range')
            # if i==int(N_axes/2):
            #     ax.set_xlabel('Best model runs')
        f1.suptitle(f"{self.BASIN} {dist} parameter clusters")
        f1.savefig(os.path.join(self.plot_folder,f'Clusters_{dist}.png'),
                    bbox_inches = 'tight')
        

    

    

    
        
        
        # parsets[key]['cluster_parsets_normed'] = cluster_parsets_normed
        # parsets[key]['pars_scaled'] = pars_scaled
        # parsets[key]['pars_df'] = pars_df






        



    