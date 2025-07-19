# Standard library imports
import os
import glob
import pickle
import warnings
import concurrent.futures
from functools import partial
from pathlib import Path
from os.path import join
from typing import Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import LightSource
from matplotlib.dates import DateFormatter
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import geopandas as gpd
from scipy.stats import pearsonr, variation, zscore, linregress
import hydrosignatures as hs
from hydrosignatures import HydroSignatures
from flexitext import flexitext
from highlight_text import fig_text

# Local imports
from SnowClass import *
from wflow_spot_setup import *
from spotpy_calib import SoilCalib
from spotpy_analysis import spotpy_analysis
from Synthetic_obs import Synthetic_obs
from Postruns import *
from Evaluation import *
from spotpy import analyser as spa
from swe_metrics import *  # Import all SWE metric functions
from q_metrics import *    # Import all Q metric functions

# Configure environment
os.environ["NUMBA_THREADING_LAYER"] = "tbb"

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=UserWarning, module='xarray')
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(invalid='ignore')

def yearly_swe_loading(year):
    print(f"Processing year {year}")
    daterange = pd.date_range(f"{year-1}-10-01", f"{year}-09-30")
    SWEyear = xr.Dataset(coords={'time': daterange})
    
    for k in range(self.SOIL_K):
        folder = join(self.YEARLYDIR, f"{year}", f"{k}", "spotpy_SWE")
        SWEfiles = glob.glob(join(folder, "*.nc"))
        cols = self.pars[self.START_YEAR].columns

        for i,f in enumerate(SWEfiles):
            if i % 100 == 0:
                print(f"Processing file {i}for year {year}")
            try:
                SWE = xr.open_dataarray(f)
                swepars = pd.DataFrame([SWE.attrs])[cols].astype(float)
                df = self.pars[year]
                difsum = np.abs(df.values - swepars.values).sum(axis=1)
                index = np.argmin(difsum)
                runid = df.index[index]
                SWEyear[runid] = SWE
            except:
                print(f"Error opening {f} with i={i}")
                break
    
    SWEyear.to_netcdf(join(self.LOA_DIR, f"{year}_SWE.nc"))
    return year, SWEyear

def yearly_calc_swe_metrics(year):
    print(f"Processing year {year}")
    year_metrics = pd.DataFrame(index=self.Q.columns)
        
    obs_signatures = self.swe_signatures[year]['obs']
    sim_signatures = self.swe_signatures[year]['sim']
    key0 = list(obs_signatures.keys())[0]
    
    for i, run_id in enumerate(self.Q.columns):
        if int(run_id.split('_')[1]) > self.runid_limit:
            year_metrics.loc[run_id, :] = np.nan
            continue
        if i%100 == 0:
            print(f"Year {year} - Processing run {i} of {len(self.Q.columns)}")
            
        # Check if we have valid simulation data for this run
        values = np.array(list(sim_signatures[key0].loc[:, run_id].values))
        if np.all(np.isnan(values)):
            year_metrics.loc[run_id, :] = np.nan
            continue
            
        # Process elevation-based metrics
        for key in obs_signatures.keys():
            if 'elev' in key:
                obs = obs_signatures[key].squeeze()
                sim = sim_signatures[key].loc[:, run_id]
                if any(substring in key for substring in ['melt_sum', 'SWE_max', 'SWE_SWS']) and not np.isin(key, 't_SWE_max_elev'):
                    # Calculate MAPE
                    try: 
                        mape = np.nanmean(np.abs((obs - sim) / obs)) * 100
                    except ZeroDivisionError:
                        mape = np.nan 
                        print(f"ZeroDivisionError for {key} in year {year} and run {run_id}")
                    year_metrics.loc[run_id, f"{key}_MAPE"] = mape
                else:
                    year_metrics.loc[run_id, f"{key}_ME"] = np.nanmean(np.abs(obs - sim))
            
            # Process grid-based metrics
            elif 'grid' in key:
                obs = obs_signatures[key]
                sim = sim_signatures[key][run_id]
                if np.isin(key, ['melt_sum_grid', 'SWE_max_grid', 'SWE_SWS_grid']) and not np.isin(key, ['t_SWE_max_grid', 't_SWE_start_grid', 't_SWE_end_grid']):
                    # Calculate MAPE
                    mape = np.nanmean(np.abs((obs - sim) / obs)) * 100
                    year_metrics.loc[run_id, f"{key}_MAPE"] = mape.item()
                else:
                    year_metrics.loc[run_id, f"{key}_ME"] = np.nanmean(np.abs(obs - sim))
            
            # Process catchment-wide metrics
            else:
                obs = obs_signatures[key]
                sim = sim_signatures[key].loc[0, run_id]
                if np.isin(key, ['melt_sum', 'SWE_max', 'SWE_SWS']) and not np.isin(key, ['t_SWE_max']):
                    # Calculate APE
                    ape = np.abs((obs - sim) / obs) * 100
                    year_metrics.loc[run_id, f"{key}_APE"] = ape
                else:
                    year_metrics.loc[run_id, f"{key}_ME"] = np.abs(obs - sim)
        
        # Calculate advanced metrics requiring original SWE data
        swe_obs = self.SWEobs.sel(time=slice(f"{year-1}-10-01", f"{year}-09-30"))
        swe_sim = self.SWE[year][run_id].sel(time=swe_obs.time)
            
            # KGE metrics
        year_metrics.loc[run_id, 'SWE_melt_KGE'] = calc_melt_kge_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_melt_KGE_elev'] = np.nanmean(
        calc_melt_kge_vs_elevation(swe_obs, swe_sim)['var'])
            
            # NSE metrics for melt
        year_metrics.loc[run_id, 'SWE_melt_NSE'] = calc_melt_nse_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_melt_NSE_elev'] = np.nanmean(
        calc_melt_nse_vs_elevation(swe_obs, swe_sim)['var'])
        year_metrics.loc[run_id, 'SWE_melt_NSE_grid'] = calc_melt_nse_grid(swe_obs, swe_sim).mean().item()
            
            # NSE metrics for snowfall
        year_metrics.loc[run_id, 'SWE_snowfall_NSE'] = calc_snowfall_nse_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_snowfall_NSE_elev'] = np.nanmean(
        calc_snowfall_nse_vs_elevation(swe_obs, swe_sim)['var'])
        year_metrics.loc[run_id, 'SWE_snowfall_NSE_grid'] = calc_snowfall_nse_grid(swe_obs, swe_sim).mean().item()
            
            # NSE metrics for SWE
        year_metrics.loc[run_id, 'SWE_NSE'] = calc_swe_nse_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_NSE_elev'] = np.nanmean(
        calc_swe_nse_vs_elevation(swe_obs, swe_sim)['var'])
        year_metrics.loc[run_id, 'SWE_NSE_grid'] = calc_swe_nse_grid(swe_obs, swe_sim).mean().item()
            
            # SPAEF metric
        year_metrics.loc[run_id, 'SWE_SPAEF'] = calculate_spaef(swe_obs, swe_sim)
    
    return year, year_metrics


def yearly_calc_swe_signatures(year):
    print(f"Processing year {year}")
    year_signatures = {}
    obs_signatures = {}
    obs_swe = self.SWEobs.sel(time=slice(f"{year-1}-10-01", f"{year}-09-30"))

    # Catchment-wide metrics
    obs_signatures['melt_sum'] = calc_melt_sum_catchment(obs_swe)
    obs_signatures['SWE_max'] = calc_swe_max_catchment(obs_swe)
    obs_signatures['SWE_SWS'] = calc_sws_catchment(obs_swe)
    obs_signatures['t_SWE_max'] = calc_t_swe_max_catchment(obs_swe)
    obs_signatures['SWE_7daymelt'] = calc_7_day_melt_catchment(obs_swe)
    obs_signatures['t_SWE_start'], obs_signatures['t_SWE_end'] = calc_swe_start_end_catchment(obs_swe)

    # Vs elevation metrics
    obs_signatures['SWE_max_elev'] = calc_swe_max_vs_elevation(obs_swe)
    obs_signatures['melt_sum_elev'] = calc_melt_sum_vs_elevation(obs_swe)
    obs_signatures['SWE_SWS_elev'] = calc_sws_vs_elevation(obs_swe)
    obs_signatures['t_SWE_max_elev'] = calc_t_swe_max_vs_elevation(obs_swe)
    obs_signatures['SWE_7daymelt_elev'] = calc_7_day_melt_vs_elevation(obs_swe)
    obs_signatures['t_SWE_start_elev'] = calc_swe_start_vs_elevation(obs_swe)
    obs_signatures['t_SWE_end_elev'] = calc_swe_end_vs_elevation(obs_swe)

    # Grid cell metrics
    obs_signatures['SWE_max_grid'] = calc_swe_max_grid(obs_swe)
    melt_sum_grid_obs = calc_melt_sum_grid(obs_swe)
    melt_sum_grid_obs = xr.where(melt_sum_grid_obs > 0, melt_sum_grid_obs, np.nan)
    obs_signatures['melt_sum_grid'] = melt_sum_grid_obs
    obs_signatures['SWE_SWS_grid'] = calc_sws_grid(obs_swe)
    obs_signatures['t_SWE_max_grid'] = calc_t_swe_max_grid(obs_swe)
    obs_signatures['SWE_7daymelt_grid'] = calc_7_day_melt_grid(obs_swe)
    obs_signatures['t_SWE_start_grid'] = calc_swe_start_grid(obs_swe)
    obs_signatures['t_SWE_end_grid'] = calc_swe_end_grid(obs_swe)

    sim_signatures = {
        'melt_sum': pd.DataFrame(columns=self.Q.columns),
        'SWE_max': pd.DataFrame(columns=self.Q.columns),
        'SWE_SWS': pd.DataFrame(columns=self.Q.columns),
        't_SWE_max': pd.DataFrame(columns=self.Q.columns),
        'SWE_7daymelt': pd.DataFrame(columns=self.Q.columns),
        't_SWE_start': pd.DataFrame(columns=self.Q.columns),
        't_SWE_end': pd.DataFrame(columns=self.Q.columns),
        'SWE_max_elev': pd.DataFrame(columns=self.Q.columns),
        'melt_sum_elev': pd.DataFrame(columns=self.Q.columns),
        'SWE_SWS_elev': pd.DataFrame(columns=self.Q.columns),
        't_SWE_max_elev': pd.DataFrame(columns=self.Q.columns),
        'SWE_7daymelt_elev': pd.DataFrame(columns=self.Q.columns),
        't_SWE_start_elev': pd.DataFrame(columns=self.Q.columns),
        't_SWE_end_elev': pd.DataFrame(columns=self.Q.columns),
        'SWE_max_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
        'melt_sum_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
        'SWE_SWS_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
        't_SWE_max_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
        'SWE_7daymelt_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
        't_SWE_start_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
        't_SWE_end_grid': xr.Dataset(coords={'lat': self.E.dem.lat, 'lon': self.E.dem.lon}),
    }

    for i, run_id in enumerate(self.Q.columns):
        if int(run_id.split('_')[1]) > self.runid_limit:
            continue
        if i % 100 == 0:
            print(f"Year {year} - Processing run {i} of {len(self.Q.columns)}")
        try:
            swe3d = self.SWE[year][run_id]
        except KeyError:
            print(f"Year {year}, run {i}: problem")
            continue

        # Catchment-wide metrics
        sim_signatures['melt_sum'].loc[0, run_id] = calc_melt_sum_catchment(swe3d)
        sim_signatures['SWE_max'].loc[0, run_id] = calc_swe_max_catchment(swe3d)
        sim_signatures['SWE_SWS'].loc[0, run_id] = calc_sws_catchment(swe3d)
        sim_signatures['t_SWE_max'].loc[0, run_id] = calc_t_swe_max_catchment(swe3d)
        sim_signatures['SWE_7daymelt'].loc[0, run_id] = calc_7_day_melt_catchment(swe3d)
        sim_signatures['t_SWE_start'].loc[0, run_id], sim_signatures['t_SWE_end'].loc[0, run_id] = calc_swe_start_end_catchment(swe3d)
        
        # Grid cell metrics
        sim_signatures['SWE_max_grid'][run_id] = calc_swe_max_grid(swe3d)
        sim_signatures['melt_sum_grid'][run_id] = calc_melt_sum_grid(swe3d)
        sim_signatures['SWE_SWS_grid'][run_id] = calc_sws_grid(swe3d)
        sim_signatures['t_SWE_max_grid'][run_id] = calc_t_swe_max_grid(swe3d)
        sim_signatures['SWE_7daymelt_grid'][run_id] = calc_7_day_melt_grid(swe3d)
        sim_signatures['t_SWE_start_grid'][run_id] = calc_swe_start_grid(swe3d)
        sim_signatures['t_SWE_end_grid'][run_id] = calc_swe_end_grid(swe3d)

        # Vs elevation metrics
        sim_signatures['SWE_max_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['SWE_max_grid'][run_id])['var']
        sim_signatures['melt_sum_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['melt_sum_grid'][run_id])['var']
        sim_signatures['SWE_SWS_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['SWE_SWS_grid'][run_id])['var']
        sim_signatures['t_SWE_max_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['t_SWE_max_grid'][run_id])['var']
        sim_signatures['SWE_7daymelt_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['SWE_7daymelt_grid'][run_id])['var']
        sim_signatures['t_SWE_start_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['t_SWE_start_grid'][run_id])['var']
        sim_signatures['t_SWE_end_elev'].loc[:, run_id] = calc_var_vs_elevation(sim_signatures['t_SWE_end_grid'][run_id])['var']

    elevation_bands = obs_signatures['SWE_max_elev'].index

    for key, value in obs_signatures.items():
        if 'elev' in key and value.shape[0] != elevation_bands.shape[0]:
            obs_signatures[key] = value.reindex(elevation_bands)
    for key, value in sim_signatures.items():
        if 'elev' in key and value.shape[0] != elevation_bands.shape[0]:
            sim_signatures[key] = value.reindex(elevation_bands)

    year_signatures['obs'] = obs_signatures
    year_signatures['sim'] = sim_signatures
    
    return year, year_signatures

def yearly_calc_Q_signatures(year, q_sim_mmd, q_obs_mmd, p_mmd, signatures):
    print(f"Processing year {year}")
    timeslice = slice(f"{year-1}-10-01", f"{year}-09-30")
    q_obs_year = q_obs_mmd.loc[timeslice]
    p_year = p_mmd.loc[timeslice]
    q_sim_year = q_sim_mmd.loc[timeslice]

    # Load the hydro signatures for observed data
    sig_obs = HydroSignatures(q_obs_year, p_year)
    
    # Precompute all signature objects for simulations
    sig_sim_dic = {}
    for run_id in self.Q.columns:
        if run_id != 'obs' and int(run_id.split('_')[1]) > self.runid_limit:
            continue
        q_sim = q_sim_year[run_id]
        sig_sim_dic[run_id] = HydroSignatures(q_sim, p_year)

        # Create a dataframe to store the results
    sig_df = pd.DataFrame(index=self.Q.columns, columns=signatures)
        
        # Process each run_id including 'obs'
    for i, run_id in enumerate(list(self.Q.columns) + ['obs']):
        if run_id != 'obs' and int(run_id.split('_')[1]) > self.runid_limit:
            continue
            
        if i % 100 == 0:
            print(f"Year {year}: Processing run {i} of {len(self.Q.columns)}")
            
        if run_id == 'obs':
            q_data = q_obs_year
        else:
            q_data = q_sim_year[run_id]
            sig_sim = sig_sim_dic[run_id]

        # Calculate basic signatures
        sig_df.loc[run_id, 'Qmax'] = np.nanmax(q_data)
        sig_df.loc[run_id, 'Qmean'] = np.nanmean(q_data)
        sig_df.loc[run_id, 'Q5'] = np.nanquantile(q_data, 0.05)
        sig_df.loc[run_id, 'Q95'] = np.nanquantile(q_data, 0.95)
        sig_df.loc[run_id, 'Qcv'] = calc_Q_cov(q_data)
        sig_df.loc[run_id, 'Qamp'] = calc_Q_amplitude(q_data)
        sig_df.loc[run_id, 't_Qmax'] = calc_t_qmax(q_data)
        sig_df.loc[run_id, 't_Qstart'] = calc_Qstart(q_data)
        sig_df.loc[run_id, 't_Qrise'] = calc_t_Qrise(q_data)

            # Calculate more complex signatures
        sig_df.loc[run_id, 't_hfd'] = calc_half_flow_date(q_data)
        sig_df.loc[run_id, 't_hfi'] = calc_half_flow_interval(q_data)
        sig_df.loc[run_id, 'high_q_freq'] = calc_high_flow_freq(q_data)
        sig_df.loc[run_id, 'peak_distribution'] = calc_peak_distribution(q_data)
        sig_df.loc[run_id, 'flashiness_index'] = calc_flashiness_index(q_data)
        sig_df.loc[run_id, 'Qcv_meltseason'] = calc_snowmeltseason_cv(q_data)
        sig_df.loc[run_id, 'Qmean_meltseason'] = calc_snowmeltseason_sum(q_data)
        sig_df.loc[run_id, 'PeakFilter_sum'] = np.nansum(self.peakfilter(q_data.values))
        sig_df.loc[run_id, 'BaseFlow_sum'] = np.nansum(self.baseflow_filter(q_data.values))
        sig_df.loc[run_id, 't_Qinflection'] = calc_Q_inflection(q_data)

        # Add hydrosignature values
        for signature in sig_obs.signature_names.keys():
            if signature in signatures:
                if run_id == 'obs':
                    sig_df.loc[run_id, signature] = sig_obs.values[signature]
                else:
                    sig_df.loc[run_id, signature] = sig_sim.values[signature]

        # Calculate baseflow recession constants
        try:
            mrc_np, bfr_k_np = hs.baseflow_recession(q_data, fit_method="nonparametric_analytic", recession_length=3)
            mrc_exp, bfr_k_exp = hs.baseflow_recession(q_data, fit_method="exponential", recession_length=3)
            sig_df.loc[run_id, 'bfr_k_np'] = bfr_k_np
            sig_df.loc[run_id, 'bfr_k_exp'] = bfr_k_exp
        except Exception as e:
            print(f"Year {year}, Error in baseflow recession for {run_id}: {e}")
            sig_df.loc[run_id, 'bfr_k_np'] = np.nan
            sig_df.loc[run_id, 'bfr_k_exp'] = np.nan

    # Return year results as a dictionary
    return year, {'obs': sig_df.loc['obs'], 'sim': sig_df.drop('obs')}

def yearly_calc_Q_metrics(year):
    print(f"Processing year {year}")
    year_metrics = pd.DataFrame(index=self.Q.columns)
        
    for i, run_id in enumerate(self.Q.columns):
        if run_id != 'obs' and int(run_id.split('_')[1]) > self.runid_limit:
            continue
            
        if i % 100 == 0:
            print(f"Year {year}: Processing run {i} of {len(self.Q.columns)}")
                
            # Get data for the water year
        q_sim = self.Q[run_id].loc[f"{year-1}-10-01":f"{year}-09-30"]
        q_obs = self.Qobs.squeeze().loc[q_sim.index]
        sig_obs = self.Q_signatures[year]['obs']
        sig_sim = self.Q_signatures[year]['sim'].loc[run_id]

        # Calculate mean error for all signatures
        for signature in sig_obs.index:
            year_metrics.loc[run_id, f"{signature}_ME"] = np.abs(sig_obs[signature] - sig_sim[signature])

        # Calculate relative errors for specified signatures
        relative_error_metrics = ['Qmean', 'Qcv', 'Q95', 'Q5', 'Qamp', 'Qmean_meltseason', 'Qcv_meltseason']
        for metric in relative_error_metrics:
            obs_value = sig_obs[metric]
            sim_value = sig_sim[metric]
            APE = np.abs((obs_value - sim_value) / obs_value) * 100
            year_metrics.loc[run_id, f"{metric}_APE"] = APE

        # Melt season data
        melt_months = [4, 5, 6, 7]
        q_sim_melt = q_sim.loc[q_sim.index.month.isin(melt_months)]
        q_obs_melt = q_obs.loc[q_obs.index.month.isin(melt_months)]
        
        # Add standard efficiency metrics
        year_metrics.loc[run_id, 'KGE'] = he.kge_2009(q_obs, q_sim)
        year_metrics.loc[run_id, 'NSE'] = he.nse(q_obs, q_sim)
        year_metrics.loc[run_id, 'R2'] = he.r_squared(q_obs, q_sim)
        year_metrics.loc[run_id, 'PBIAS'] = np.abs((q_sim.sum() - q_obs.sum()) / q_obs.sum()) * 100
        
        # Filtered metrics
        year_metrics.loc[run_id, 'KGE_PeakFilter'] = he.kge_2009(
            self.peakfilter(q_sim.values), 
            self.peakfilter(q_obs.values)
        )
        year_metrics.loc[run_id, 'KGE_BaseFlow'] = he.kge_2009(
            self.baseflow_filter(q_sim.values), 
            self.baseflow_filter(q_obs.values)
        )
        
        # Melt season metrics
        year_metrics.loc[run_id, 'NSE_meltseason'] = he.nse(q_sim_melt, q_obs_melt)
        year_metrics.loc[run_id, 'logNSE'] = he.nse(np.log(q_sim), np.log(q_obs))
        year_metrics.loc[run_id, 'logNSE_meltseason'] = he.nse(np.log(q_sim_melt), np.log(q_obs_melt))

        # Three components of KGE for melt season
        r, alpha, beta, kge = he.kge_2009(q_sim_melt, q_obs_melt, return_all=True)
        year_metrics.loc[run_id, 'KGE_meltseason_r'] = r
        year_metrics.loc[run_id, 'KGE_meltseason_alpha'] = alpha
        year_metrics.loc[run_id, 'KGE_meltseason_beta'] = beta
        year_metrics.loc[run_id, 'KGE_meltseason'] = kge
        year_metrics.loc[run_id, 'logKGE_meltseason'] = he.kge_2009(np.log(q_sim_melt), np.log(q_obs_melt))
        
    return year, year_metrics
def load_swe_file(year, loa_dir):
    return year, xr.open_dataset(join(loa_dir, f"{year}_SWE.nc"))

class LOA:
    """
    Class for the LOA analysis.
    """
    def __init__(self, config):
        """
        Initialize the LOA class.
        Parameters:
        cfg (dict): The configuration dictionary.
        Returns:
        None
        This function initializes the LOA class with the configuration dictionary.
        """
        self.config  = config
        self.E = Evaluation(config)

        for key, value in config.items():
            setattr(self, key, value)

        #create dir for output
        self.LOA_DIR = join(self.OUTDIR, 'LOA')
        Path.mkdir(Path(self.LOA_DIR), exist_ok=True)

    def load_Q_obs(self):
        self.E.load_Q_Synthetic()
        qobs = self.E.Q_Synthetic
        self.Qobs = qobs
    def load_SWE_obs(self):
        
        sweobs = self.E.load_SWE_Synthetic()
        self.SWEobs = sweobs
    def load_spotpy_analyses(self):
        self.SA = {}
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            for ksoil in range(self.SOIL_K):
                RUN_NAME = f"{config['BASIN']}_{year}_ms{ksoil}_{config['EXP_ID']}"
                SA = spotpy_analysis(config, RUN_NAME, 'Yearlycalib', single_year = year, member = ksoil)
                SA._set_attributes_from_config()
                SA._initialize_paths_and_files()
                SA._load_results()
                # SA._load_Q()
                SA.Q = pd.DataFrame(spa.get_modelruns(SA.results)).transpose()
                self.SA[(year, ksoil)] = SA

    def load_Q(self):
        Qlist = []
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            daterange = pd.date_range(f"{year-1}-10-01",f"{year}-09-30")
            Qyear= pd.DataFrame(index = daterange)
            for ksoil in range(self.SOIL_K):
                Q = self.SA[(year, ksoil)].Q.copy()
                prefix = f"ms{ksoil}"
                Q.columns = [f"{prefix}_{col}" for col in Q.columns]
                Q.index= daterange
                Qyear = pd.concat([Qyear,Q],axis=1)
            Qlist.append(Qyear)
        self.Q = pd.concat(Qlist)
    
    def load_pars(self):
        parsdic = {}
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            parsdic[year] = []
            for ksoil in range(self.SOIL_K):
                SA = self.SA[(year, ksoil)]
                pars = pd.DataFrame(SA.params)
                pars.columns = pars.columns.str.replace('par','')
                pars.index = [f"ms{ksoil}_{i}" for i in np.arange(len(pars))]
                parsdic[year].append(pars)
            parsdic[year] = pd.concat(parsdic[year])
        self.pars = parsdic        


    def load_spotpy_SWE(self):
        # print('loading SWE and extracting parameters')
        import concurrent.futures

        from concurrent.futures import ProcessPoolExecutor
        SWEfiles = [join(self.LOA_DIR,f"{year}_SWE.nc") for year in range(self.START_YEAR,self.END_YEAR+1)]
        if np.all([os.path.isfile(f) for f in SWEfiles]):
            print('loading SWE from files')
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(load_swe_file, year, self.LOA_DIR) for year in range(self.START_YEAR, self.END_YEAR + 1)]
                SWEdic = dict(f.result() for f in futures)

            self.SWE = SWEdic
        else:
            print('loading SWE from spotpy')
            parallel_LOA = True
            SWEdic = {}
            cols = self.pars[self.START_YEAR].columns
              
            # Process years in parallel using ProcessPoolExecutor
            # (ThreadPoolExecutor might be better if I/O bound)
            years = list(range(self.START_YEAR, self.END_YEAR + 1))
            num_batches = 2  # You can adjust this number based on your needs
            batch_size = max(1, len(years) // num_batches)
            max_workers = min(int(os.cpu_count()//2), len(years))  # More threads than cores can be beneficial for I/O bound tasks
            if parallel_LOA:

                for i in range(0, len(years), batch_size):
                    batch = years[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{num_batches}: years {batch}")
                    
                    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
                        batch_results = list(executor.map(yearly_swe_loading, batch))


                    # Add batch results to dictionary
                    for year, SWEyear in batch_results:
                        SWEdic[year] = SWEyear
                    # for year, SWEyear in results:
                    #     SWEdic[year] = SWEyear
            else:
                for year in years:
                    print(f"Processing year {year}")
                    # Load SWE for the current year
                    swe_year = yearly_swe_loading(year)
                    # Store the result in the dictionary
                    SWEdic[year] = swe_year
            self.SWE = SWEdic
          

    def calc_swe_signatures(self):
        """
        Calculate SWE signatures and store them in self.swe_signatures.
        Uses ThreadPoolExecutor for parallel processing across years.
        """
        import concurrent.futures
        from functools import partial
        
        swe_indices_file = join(self.LOA_DIR, 'swe_indices.pkl')
        if os.path.isfile(swe_indices_file):
            print('Loading SWE indices from file')
            with open(swe_indices_file, 'rb') as f:
                self.swe_signatures = pickle.load(f)
            return

        if not hasattr(self, 'SWE'):
            self.load_spotpy_SWE()

        print("Calculating SWE signatures with ThreadPoolExecutor")
        years = list(range(self.START_YEAR, self.END_YEAR + 1))
        self.swe_signatures = {}
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(int(os.cpu_count()//2), len(years))  # More threads than cores can be beneficial for I/O bound tasks
        process_year_partial = partial(yearly_calc_swe_signatures)
        
        time0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and create a future-to-year mapping
            future_to_year = {executor.submit(process_year_partial, year): year for year in years}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    result_year, year_signatures = future.result()
                    self.swe_signatures[result_year] = year_signatures
                    print(f"Year {result_year} processing completed")
                except Exception as exc:
                    print(f"Year {year} generated an exception: {exc}")
        print(f"All years processed in {time.time() - time0:.2f} seconds")
        # Save signatures to file
        print("Saving SWE signatures to file")
        with open(swe_indices_file, 'wb') as f:
            pickle.dump(self.swe_signatures, f)
                
    def calc_swe_metrics(self):
        """
        Calculate SWE metrics and store them in self.swemetrics.
        Processes years in parallel for improved performance.
        """
        print('Calculating SWE metrics')
        if not hasattr(self, 'swe_signatures'):
            print('Calculating SWE signatures first')
            self.calc_swe_signatures()

        swemetrics_file = join(self.LOA_DIR, 'swemetrics.pkl')
        if os.path.isfile(swemetrics_file):
            print('Loading SWE metrics from file')
            with open(swemetrics_file, 'rb') as f:
                self.swemetrics = pickle.load(f)
            return

        print('Calculating SWE metrics in parallel')
        
        # Initialize the metrics dictionary
        self.swemetrics = {year: pd.DataFrame(index=self.Q.columns) 
                        for year in range(self.START_YEAR, self.END_YEAR + 1)}
        
        # Function to process a single year
        years = list(range(self.START_YEAR, self.END_YEAR + 1))
        workers = min(int(os.cpu_count()//2), len(years))  # More threads than cores can be beneficial for I/O bound tasks
        
        # Process years in parallel
        import concurrent.futures
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all years for processing
            future_to_year = {
                executor.submit(yearly_calc_swe_metrics, year): year 
                for year in years
            }
            
            # Collect results as they become available
            for future in concurrent.futures.as_completed(future_to_year):
                year, metrics = future.result()
                self.swemetrics[year] = metrics
                print(f"Completed metrics for year {year}")
        
        # Save the final results
        with open(swemetrics_file, 'wb') as f:
            pickle.dump(self.swemetrics, f)
        
        print("All SWE metrics calculated and saved.")
            
    def load_meteo(self):
        meteofiles = glob.glob(join(self.CONTAINER,f"wflow_MeteoSwiss_*.nc"))
        meteo_basefile = join(self.CONTAINER,f"wflow_MeteoSwiss_{self.RESOLUTION}_{self.BASIN}_{self.START_YEAR-2}_{self.END_YEAR}.nc")
        self.meteo_base = xr.open_dataset(meteo_basefile)

    def calc_Q_signatures(self):
        """
        Calculate Q signatures and store them in self.Q_signatures.
        Uses ThreadPoolExecutor for parallel processing across years.
        """
        print('Calculating Q signatures')
        import hydrosignatures as hs
        from hydrosignatures import HydroSignatures
        from functools import partial
        import concurrent.futures

        Q_signatures_file = join(self.LOA_DIR, 'Q_signatures.pkl')
        if os.path.isfile(Q_signatures_file):
            print('Loading Q signatures from file')
            with open(Q_signatures_file, 'rb') as f:
                self.Q_signatures = pickle.load(f)
            return

        # Convert streamflow from m³/s to mm/day
        q_sim_mmd = m3s_to_mm(self.Q, self.E.dem_area)
        q_obs_mmd = m3s_to_mm(self.Qobs, self.E.dem_area).squeeze()
        p_mmd = self.meteo_base['pr'].mean(dim=['lat', 'lon']).loc[q_sim_mmd.index].to_pandas()

        # List of signatures to calculate
        signatures = [
            'Qmax', 'Qmean', 'Q5', 'Q95', 'Qcv', 'Qamp', 't_Qmax', 't_hfd', 't_hfi', 't_Qrise', 't_Qstart',
            'high_q_freq', 'bfi', 'runoff_ratio', 'fdc_slope', 'flashiness_index', 'bfr_k_np', 'bfr_k_exp',
            'peak_distribution', 'Qcv_meltseason', 'Qmean_meltseason', 'PeakFilter_sum', 'BaseFlow_sum', 't_Qinflection'
        ]

        # Function to process a single year
        

        # Initialize results dictionary
        self.Q_signatures = {}
        
        # Set up the partial function with common arguments
        process_year_partial = partial(
            yearly_calc_Q_signatures, 
            q_sim_mmd=q_sim_mmd, 
            q_obs_mmd=q_obs_mmd, 
            p_mmd=p_mmd, 
            signatures=signatures
        )
        
        # Get list of years to process
        years = list(range(self.START_YEAR, self.END_YEAR + 1))
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(int(os.cpu_count()//2), len(years))  # More threads than cores can be beneficial for I/O bound tasks
        print(f"Processing {len(years)} years with {max_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and create a future-to-year mapping
            future_to_year = {executor.submit(process_year_partial, year): year for year in years}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_result, signatures_dict = future.result()
                    self.Q_signatures[year_result] = signatures_dict
                    print(f"Completed processing for year {year_result}")
                except Exception as e:
                    print(f"Error processing year {year}: {e}")
        
        # Save signatures to file
        print("Saving Q signatures to file")
        with open(Q_signatures_file, 'wb') as f:
            pickle.dump(self.Q_signatures, f)
        
        print("Q signatures calculation completed")
    
    def calc_Q_metrics(self):
        """
        Calculate Q metrics and store them in self.Qmetrics.
        Processes years in parallel for improved performance.
        """
        print('Calculating Q metrics')
        if not hasattr(self, 'Q_signatures'):
            self.calc_Q_signatures()

        Qmetrics_file = join(self.LOA_DIR, 'Qmetrics.pkl')
        if os.path.isfile(Qmetrics_file):
            print('Loading Q metrics from file')
            with open(Qmetrics_file, 'rb') as f:
                self.Qmetrics = pickle.load(f)
            return

        print('Calculating Q metrics in parallel')
        
        # Initialize metrics dictionary for all years
        self.Qmetrics = {year: pd.DataFrame(index=self.Q.columns) 
                        for year in range(self.START_YEAR, self.END_YEAR + 1)}
        
        # Function to process a single year
        # @staticmethod
        
        
        # Process all years in parallel
        import concurrent.futures
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all years for processing
            future_to_year = {
                executor.submit(yearly_calc_Q_metrics, year): year 
                for year in range(self.START_YEAR, self.END_YEAR + 1)
            }
            
            # Collect results as they become available
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_result, year_metrics = future.result()
                    self.Qmetrics[year_result] = year_metrics
                    print(f"Completed processing for year {year_result}")
                except Exception as exc:
                    print(f"Year {year} generated an exception: {exc}")
        
        # Save the final results
        with open(Qmetrics_file, 'wb') as f:
            pickle.dump(self.Qmetrics, f)
        
        print("All Q metrics calculated and saved.")

    def combine_metrics(self):
        print('Combining metrics')
        if not hasattr(self,'metrics'):
            self.metrics = {}
            for year in range(self.START_YEAR, self.END_YEAR+1):
                self.metrics[year] = pd.DataFrame(index = self.Q.columns)
        for year in range(self.START_YEAR, self.END_YEAR+1):
            self.metrics[year] = pd.concat([self.Qmetrics[year],self.swemetrics[year]],axis=1)
    def save_metrics(self):
        print('Saving metrics')
        for year in range(self.START_YEAR, self.END_YEAR+1):
            metrics_file = join(self.LOA_DIR, f"{year}_metrics.csv")
            self.metrics[year].to_csv(metrics_file)
    def load_metrics(self):

        print('Loading metrics')
        
        # Initialize the metrics attribute if it doesn't exist
        if not hasattr(self, 'metrics'):
            self.metrics = {}
        
            # Load metrics from files
            for year in range(self.START_YEAR, self.END_YEAR + 1):
                metrics_file = join(self.LOA_DIR, f"{year}_metrics.csv")
                print(metrics_file)
                
                if os.path.isfile(metrics_file):
                    self.metrics[year] = pd.read_csv(metrics_file, index_col=0)
                else:
                    print(f'No metrics file found for year {year}')
                    break
    
    
#%%
if __name__ == '__main__':
    config_dir = "/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files"
    # # config_dir="/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle/experiments/config_files"

    # EXP_ID = 'Syn_032b'
    # EXP_ID = 'Syn_162LBa'
    # EXPS = ['Syn_34b','Syn_162LBa']#,'Syn_162LBa']¨
    # EXPS = ['Syn_244a','Syn_244b','Syn_162LBa']
    # EXPS = ['Syn_162b','Syn_162LB']
    # EXPS = ['Syn_162a','Syn_162LBa']
    # EXP_ID = 'Syn_162a'
    # EXP_ID = 'Syn_241'
    # EXP_ID = 'Syn_261plus2'
    EXP_ID = 'Syn_912a'
    plotting = True
    # EXPS = ['Syn_032b','Syn_042']#,'Syn_0425']#,'Syn_2915']
    EXPS = [EXP_ID]#'Syn_2911',
    # EXPS = {'Syn_2911':{},
    #         'Syn_29102':{},
    #         'Syn_2915':{}}
    LOA_objects = {}
    for EXP_ID in EXPS:
        print(EXP_ID)
        ORIG_ID = EXP_ID
        if 'LBa' in EXP_ID:
            EXP_ID = EXP_ID.replace('LBa','LB')
        if 'bb' in EXP_ID:
            EXP_ID = EXP_ID.replace('bb','b')
        if EXP_ID == 'Syn_244b':
            EXP_ID = 'Syn_244a'

        cfg_file = join(config_dir, f"{EXP_ID}_config.json")
        with open(cfg_file, 'r') as f:
            config = json.load(f)
        # config['BASIN'] = "Riale_di_Calneggia"
        config['BASIN'] = 'Dischma'
        for key,value in config.items():
            #change the paths to the correct paths
            if type(value)==str and '/work/FAC' in value:
                config[key] = value.replace('/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle',
                '/home/pwiersma/scratch/Data/ewatercycle')

        if 'LBa' in ORIG_ID:
            print('Lwer benchmark a')
            config['SYNDIR'] = config['SYNDIR'].replace('Synthetic_obs','Synthetic_obs_a')
            for f in glob.glob(config['SYNDIR']+'/*'):
                if '162a' in f:
                    shutil.copy(f,f.replace('162a','162LB'))
            config['SYN_SNOWMODEL'] = 'Hock'
            config['OUTDIR'] = config['OUTDIR'].replace('162LB','162LBa')
        
        if '162a' in ORIG_ID: 
            config['SYNDIR'] = config['SYNDIR'].replace('Synthetic_obs','Synthetic_obs_182a')
            for f in glob.glob(config['SYNDIR']+'/*'):
                print(f)
                filename = os.path.basename(f)
                if '182a' in filename:
                    print('changing file')
                    new_filename = filename.replace('182a', '162a')
                    new_filepath = os.path.join(os.path.dirname(f), new_filename)
                    shutil.copy(f, new_filepath)


        if '162b' in ORIG_ID:
            config['SYNDIR'] = config['SYNDIR'].replace('Synthetic_obs','Synthetic_obs_182b')
            for f in glob.glob(config['SYNDIR']+'/*'):
                print(f)
                filename = os.path.basename(f)
                if '182b' in filename:
                    print('changing file')
                    new_filename = filename.replace('182b', '162b')
                    new_filepath = os.path.join(os.path.dirname(f), new_filename)
                    shutil.copy(f, new_filepath)
       
        self = LOA(config)
        if 'LB' in EXP_ID:
            self.runid_limit = 1400
        else:
            self.runid_limit = 1e6
        self.runid_limit = 5000
        print(f"Runid_limit = {self.runid_limit}")


        if (ORIG_ID =='Syn_34bb') or (ORIG_ID == 'Syn_244b'):
            print('Changing to OSHD')
            self.SYNDIR = self.SYNDIR.replace('Synthetic_obs','Synthetic_obs_OSHD')
            self.E.SYNDIR = self.SYNDIR 
            self.SYN_SNOWMODEL = 'OSHD'
            self.E.SYN_SNOWMODEL = 'OSHD'
            self.LOA_DIR = self.LOA_DIR.replace('LOA','LOA_OSHD')
            Path(self.LOA_DIR).mkdir(parents=True, exist_ok=True)
            print(self.E.SYN_SNOWMODEL, self.SYN_SNOWMODEL, self.LOA_DIR)
        self.ORIG_ID = ORIG_ID

        self.FIGDIR = join(self.LOA_DIR, 'Figures')
        if not os.path.isdir(self.FIGDIR):
            os.makedirs(self.FIGDIR)
        # import cProfile 
        # import pstats
        # profiler = cProfile.Profile()
        # profiler.enable()
        
        self.load_spotpy_analyses()
        self.load_Q_obs()
        self.Qobs.index = pd.to_datetime(self.Qobs.index, format = '%d-%m-%Y')
        print(self.Qobs)
        self.load_SWE_obs()
        self.load_Q()
        self.load_pars()
        swe_t = self.SWEobs.sel(time = slice('2020-10','2021-09')).mean(dim = ['lat','lon'])
        print(swe_t.mean().item())

        self.elev_bands=calc_elev_bands(self.E.dem, N_bands = 10)
        self.E.bands = list(self.elev_bands)
        self.load_spotpy_SWE()
        self.calc_swe_signatures()
        self.calc_swe_metrics()
        self.load_meteo()
        self.calc_Q_signatures()
        self.calc_Q_metrics()
        self.combine_metrics()
        self.save_metrics()