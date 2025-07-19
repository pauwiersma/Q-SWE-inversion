import os
# Add at the top of your script, before any Numba imports
import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
import pandas as pd
from os.path import join
from SnowClass import *
from wflow_spot_setup import *
# from hydrographs_fromdic import *
import seaborn as sns
import matplotlib.pyplot as plt
from flexitext import flexitext
from matplotlib import patches
# from pypalettes import load_cmap
from rasterio.mask import mask
from matplotlib.colors import LightSource
from rasterio.plot import show
import math
import rasterio 
import geopandas as gpd
from typing import Tuple,Union
from matplotlib.dates import DateFormatter 
import glob
from highlight_text import fig_text


from spotpy_calib import SoilCalib
from spotpy_analysis import spotpy_analysis
from Synthetic_obs import Synthetic_obs
from Postruns import *
from Evaluation import *
from spotpy import analyser as spa
import pickle
import hydrosignatures as hs
from hydrosignatures import HydroSignatures
from functools import partial
import concurrent.futures

from scipy.stats import pearsonr, variation
from scipy.stats import zscore
#import linregress
from scipy.stats import linregress

#surpress warning
# pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
np.seterr(invalid='ignore')
#surpress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='xarray')
warnings.filterwarnings("ignore", category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
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
        year_metrics.loc[run_id, 'SWE_melt_KGE'] = self.calc_melt_kge_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_melt_KGE_elev'] = np.nanmean(
        self.calc_melt_kge_vs_elevation(swe_obs, swe_sim)['var'])
            
            # NSE metrics for melt
        year_metrics.loc[run_id, 'SWE_melt_NSE'] = self.calc_melt_nse_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_melt_NSE_elev'] = np.nanmean(
        self.calc_melt_nse_vs_elevation(swe_obs, swe_sim)['var'])
        year_metrics.loc[run_id, 'SWE_melt_NSE_grid'] = self.calc_melt_nse_grid(swe_obs, swe_sim).mean().item()
            
            # NSE metrics for snowfall
        year_metrics.loc[run_id, 'SWE_snowfall_NSE'] = self.calc_snowfall_nse_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_snowfall_NSE_elev'] = np.nanmean(
        self.calc_snowfall_nse_vs_elevation(swe_obs, swe_sim)['var'])
        year_metrics.loc[run_id, 'SWE_snowfall_NSE_grid'] = self.calc_snowfall_nse_grid(swe_obs, swe_sim).mean().item()
            
            # NSE metrics for SWE
        year_metrics.loc[run_id, 'SWE_NSE'] = self.calc_swe_nse_catchment(swe_obs, swe_sim)
        year_metrics.loc[run_id, 'SWE_NSE_elev'] = np.nanmean(
        self.calc_swe_nse_vs_elevation(swe_obs, swe_sim)['var'])
        year_metrics.loc[run_id, 'SWE_NSE_grid'] = self.calc_swe_nse_grid(swe_obs, swe_sim).mean().item()
            
            # SPAEF metric
        year_metrics.loc[run_id, 'SWE_SPAEF'] = self.calculate_spaef(swe_obs, swe_sim)
    
    return year, year_metrics


def yearly_calc_swe_signatures(year):
    print(f"Processing year {year}")
    year_signatures = {}
    obs_signatures = {}
    obs_swe = self.SWEobs.sel(time=slice(f"{year-1}-10-01", f"{year}-09-30"))

    # Catchment-wide metrics
    obs_signatures['melt_sum'] = self.calc_melt_sum_catchment(obs_swe)
    obs_signatures['SWE_max'] = self.calc_swe_max_catchment(obs_swe)
    obs_signatures['SWE_SWS'] = self.calc_sws_catchment(obs_swe)
    obs_signatures['t_SWE_max'] = self.calc_t_swe_max_catchment(obs_swe)
    obs_signatures['SWE_7daymelt'] = self.calc_7_day_melt_catchment(obs_swe)
    obs_signatures['t_SWE_start'], obs_signatures['t_SWE_end'] = self.calc_swe_start_end_catchment(obs_swe)

    # Vs elevation metrics
    obs_signatures['SWE_max_elev'] = self.calc_swe_max_vs_elevation(obs_swe)
    obs_signatures['melt_sum_elev'] = self.calc_melt_sum_vs_elevation(obs_swe)
    obs_signatures['SWE_SWS_elev'] = self.calc_sws_vs_elevation(obs_swe)
    obs_signatures['t_SWE_max_elev'] = self.calc_t_swe_max_vs_elevation(obs_swe)
    obs_signatures['SWE_7daymelt_elev'] = self.calc_7_day_melt_vs_elevation(obs_swe)
    obs_signatures['t_SWE_start_elev'] = self.calc_swe_start_vs_elevation(obs_swe)
    obs_signatures['t_SWE_end_elev'] = self.calc_swe_end_vs_elevation(obs_swe)

    # Grid cell metrics
    obs_signatures['SWE_max_grid'] = self.calc_swe_max_grid(obs_swe)
    melt_sum_grid_obs = self.calc_melt_sum_grid(obs_swe)
    melt_sum_grid_obs = xr.where(melt_sum_grid_obs > 0, melt_sum_grid_obs, np.nan)
    obs_signatures['melt_sum_grid'] = melt_sum_grid_obs
    obs_signatures['SWE_SWS_grid'] = self.calc_sws_grid(obs_swe)
    obs_signatures['t_SWE_max_grid'] = self.calc_t_swe_max_grid(obs_swe)
    obs_signatures['SWE_7daymelt_grid'] = self.calc_7_day_melt_grid(obs_swe)
    obs_signatures['t_SWE_start_grid'] = self.calc_swe_start_grid(obs_swe)
    obs_signatures['t_SWE_end_grid'] = self.calc_swe_end_grid(obs_swe)

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
        sim_signatures['melt_sum'].loc[0, run_id] = self.calc_melt_sum_catchment(swe3d)
        sim_signatures['SWE_max'].loc[0, run_id] = self.calc_swe_max_catchment(swe3d)
        sim_signatures['SWE_SWS'].loc[0, run_id] = self.calc_sws_catchment(swe3d)
        sim_signatures['t_SWE_max'].loc[0, run_id] = self.calc_t_swe_max_catchment(swe3d)
        sim_signatures['SWE_7daymelt'].loc[0, run_id] = self.calc_7_day_melt_catchment(swe3d)
        sim_signatures['t_SWE_start'].loc[0, run_id], sim_signatures['t_SWE_end'].loc[0, run_id] = self.calc_swe_start_end_catchment(swe3d)
        
        # Grid cell metrics
        sim_signatures['SWE_max_grid'][run_id] = self.calc_swe_max_grid(swe3d)
        sim_signatures['melt_sum_grid'][run_id] = self.calc_melt_sum_grid(swe3d)
        sim_signatures['SWE_SWS_grid'][run_id] = self.calc_sws_grid(swe3d)
        sim_signatures['t_SWE_max_grid'][run_id] = self.calc_t_swe_max_grid(swe3d)
        sim_signatures['SWE_7daymelt_grid'][run_id] = self.calc_7_day_melt_grid(swe3d)
        sim_signatures['t_SWE_start_grid'][run_id] = self.calc_swe_start_grid(swe3d)
        sim_signatures['t_SWE_end_grid'][run_id] = self.calc_swe_end_grid(swe3d)

        # Vs elevation metrics
        sim_signatures['SWE_max_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['SWE_max_grid'][run_id])['var']
        sim_signatures['melt_sum_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['melt_sum_grid'][run_id])['var']
        sim_signatures['SWE_SWS_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['SWE_SWS_grid'][run_id])['var']
        sim_signatures['t_SWE_max_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['t_SWE_max_grid'][run_id])['var']
        sim_signatures['SWE_7daymelt_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['SWE_7daymelt_grid'][run_id])['var']
        sim_signatures['t_SWE_start_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['t_SWE_start_grid'][run_id])['var']
        sim_signatures['t_SWE_end_elev'].loc[:, run_id] = self.calc_var_vs_elevation(sim_signatures['t_SWE_end_grid'][run_id])['var']

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
        sig_df.loc[run_id, 'Qcv'] = self.calc_Q_cov(q_data)
        sig_df.loc[run_id, 'Qamp'] = self.calc_Q_amplitude(q_data)
        sig_df.loc[run_id, 't_Qmax'] = self.calc_t_qmax(q_data)
        sig_df.loc[run_id, 't_Qstart'] = self.calc_Qstart(q_data)
        sig_df.loc[run_id, 't_Qrise'] = self.calc_t_Qrise(q_data)

            # Calculate more complex signatures
        sig_df.loc[run_id, 't_hfd'] = self.calc_half_flow_date(q_data)
        sig_df.loc[run_id, 't_hfi'] = self.calc_half_flow_interval(q_data)
        sig_df.loc[run_id, 'high_q_freq'] = self.calc_high_flow_freq(q_data)
        sig_df.loc[run_id, 'peak_distribution'] = self.calc_peak_distribution(q_data)
        sig_df.loc[run_id, 'flashiness_index'] = self.calc_flashiness_index(q_data)
        sig_df.loc[run_id, 'Qcv_meltseason'] = self.calc_snowmeltseason_cv(q_data)
        sig_df.loc[run_id, 'Qmean_meltseason'] = self.calc_snowmeltseason_sum(q_data)
        sig_df.loc[run_id, 'PeakFilter_sum'] = np.nansum(self.peakfilter(q_data.values))
        sig_df.loc[run_id, 'BaseFlow_sum'] = np.nansum(self.baseflow_filter(q_data.values))
        sig_df.loc[run_id, 't_Qinflection'] = self.calc_Q_inflection(q_data)

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
def translate_SWE_metric_name(metric_name,keep_suffix = True):
    new_name = metric_name
    elev = False
    grid = False
    if 'elev' in metric_name:
        new_name = metric_name.replace('_elev', '')
        elev = True
    elif 'grid' in metric_name:
        new_name = metric_name.replace('_grid', '')
        grid = True
        
    readable_dic = {
        'melt_sum': 'Accumulation',
        'SWE_melt_KGE': 'Melt',
        'SWE_melt_NSE': 'Melt',
        'SWE_NSE': 'SWE',
        'SWE_SWS': 'SWS',
        't_SWE_start': 'Onset',#'Appearance',#'Start of SWE timing',
        't_SWE_end': 'Melt-out',#'Disappearance',#'End of SWE timing',
        't_SWE_max': 'Peak timing',#'Maximum SWE timing',
        'SWE_snowfall_NSE': 'Snowfall',
        'SWE_SPAEF': 'Spatial Efficiency'  # Fixed this line
    }
    for key in readable_dic.keys():
        if key in new_name:
            # print(key)
            new_name = new_name.replace(key, readable_dic[key])
    if '_' in new_name:
        new_name = new_name.replace('_'+new_name.split('_')[-1], ' ')
    if keep_suffix:
        if elev:
            new_name = 'ELEV-' + new_name
        elif grid:
            new_name = 'GRID-' + new_name
        else:
            new_name = 'AGG-' + new_name
    # if '_' in new_name:
    #     # new_name = new_name.replace('_', ' ')
    #     new_name = new_name.split('_')[0]
    return new_name
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

    # Sum of SWE
    def calc_sum2d(self, swe3d):
        """
        Calculate the sum of SWE over latitude and longitude, normalized by the number of valid cells.
        """
        sum2d = swe3d.sum(dim=['lat', 'lon'])
        cell_count = np.sum(~np.isnan(self.E.dem))
        sum2d = sum2d / cell_count
        return sum2d

    # Maximum SWE timing
    def calc_t_swe_max(self, swe_t):
        """
        Calculate the time of maximum SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        t_swe_max = np.nanargmax(swe_t)
        return t_swe_max if not np.isnan(t_swe_max) else np.nan

    def calc_t_swe_max_vs_elevation(self, swe3d):
        """
        Calculate the time of maximum SWE for different elevation bands.
        """
        t_swe_max = xr.apply_ufunc(self.calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        t_swe_max_vs_elevation = self.calc_var_vs_elevation(t_swe_max)
        return t_swe_max_vs_elevation

    def calc_t_swe_max_catchment(self, swe3d):
        """
        Calculate the time of maximum SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        t_swe_max = self.calc_t_swe_max(swe_t)
        return t_swe_max

    def calc_t_swe_max_grid(self, swe3d):
        """
        Calculate the time of maximum SWE for each grid cell.
        """
        t_swe_max_grid = xr.apply_ufunc(self.calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        return t_swe_max_grid
    
    # Maximum SWE
    def calc_swe_max(self, swe_t):
        """
        Calculate the maximum SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        SWE_max = np.nanmax(swe_t)
        return SWE_max
    
    def calc_swe_max_catchment(self, swe3d):
        """
        Calculate the maximum SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        SWE_max = self.calc_swe_max(swe_t)
        return SWE_max
    
    def calc_swe_max_vs_elevation(self, swe3d):
        """
        Calculate the maximum SWE for different elevation bands.
        """
        SWE_max = xr.apply_ufunc(self.calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        SWE_max_vs_elevation = self.calc_var_vs_elevation(SWE_max)
        return SWE_max_vs_elevation

    def calc_swe_max_grid(self, swe3d):
        """
        Calculate the maximum SWE for each grid cell.
        """
        SWE_max_grid = xr.apply_ufunc(self.calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
        return SWE_max_grid

    # Snow Water Storage (SWS)
    def calc_sws(self, swe_t):
        """
        Calculate Snow Water Storage (SWS), the area under the SWE curve.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        sws = np.nansum(swe_t)
        return sws

    def calc_sws_vs_elevation(self, swe3d):
        """
        Calculate Snow Water Storage (SWS) for different elevation bands.
        """
        sws = xr.apply_ufunc(self.calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)
        sws_vs_elevation = self.calc_var_vs_elevation(sws)
        return sws_vs_elevation

    def calc_sws_catchment(self, swe3d):
        """
        Calculate the total Snow Water Storage (SWS) for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        sws = self.calc_sws(swe_t)
        return sws

    def calc_sws_grid(self, swe3d):
        """
        Calculate Snow Water Storage (SWS) for each grid cell.
        """
        sws_grid = xr.apply_ufunc(self.calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)
        return sws_grid

    # Melt Rate
    def calc_melt_rate(self, swe_t):
        """
        Calculate the mean melt rate of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        diff = np.diff(swe_t)
        neg_diff = diff[diff < 0]
        mean_melt_rate = np.mean(neg_diff) * -1
        return mean_melt_rate

    def calc_melt_rate_vs_elevation(self, swe3d):
        """
        Calculate the mean melt rate of SWE for different elevation bands.
        """
        melt_rate = xr.apply_ufunc(self.calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)
        melt_rate_vs_elevation = self.calc_var_vs_elevation(melt_rate)
        return melt_rate_vs_elevation

    def calc_melt_rate_catchment(self, swe3d):
        """
        Calculate the mean melt rate of SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        melt_rate = self.calc_melt_rate(swe_t)
        return melt_rate

    def calc_melt_rate_grid(self, swe3d):
        """
        Calculate the mean melt rate of SWE for each grid cell.
        """
        melt_rate_grid = xr.apply_ufunc(self.calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)
        return melt_rate_grid

    # 7-Day Melt Rate
    def calc_7_day_melt(self, swe_t):
        """
        Calculate the maximum 7-day melt rate of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        diff = np.diff(swe_t)
        neg_diff = diff[diff < 0] * -1
        rol7 = pd.Series(neg_diff).rolling(window=7).max().max()
        return rol7

    def calc_7_day_melt_vs_elevation(self, swe3d):
        """
        Calculate the maximum 7-day melt rate of SWE for different elevation bands.
        """
        melt7 = xr.apply_ufunc(self.calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)
        melt7_vs_elevation = self.calc_var_vs_elevation(melt7)
        return melt7_vs_elevation

    def calc_7_day_melt_catchment(self, swe3d):
        """
        Calculate the maximum 7-day melt rate of SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        melt7 = self.calc_7_day_melt(swe_t)
        return melt7

    def calc_7_day_melt_grid(self, swe3d):
        """
        Calculate the maximum 7-day melt rate of SWE for each grid cell.
        """
        melt7_grid = xr.apply_ufunc(self.calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)
        return melt7_grid

    # Total SWE
    def calc_melt_sum(self, swe_t):
        """
        Calculate the total SWE.
        """
        diff = np.diff(swe_t)
        pos_swe = diff[diff < 0]
        melt_sum = np.nansum(pos_swe)*-1
        return melt_sum
    
    def calc_sf_sum(self, swe_t):
        """
        Calculate the total SWE.
        """
        diff = np.diff(swe_t)
        pos_swe = diff[diff > 0]
        melt_sum = np.nansum(pos_swe)
        return melt_sum

    def calc_melt_sum_vs_elevation(self, swe3d):
        """
        Calculate the total SWE for different elevation bands.
        """
        melt_sum = xr.apply_ufunc(self.calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
        melt_sum = xr.where(melt_sum > 0, melt_sum, np.nan)
        melt_sum_vs_elevation = self.calc_var_vs_elevation(melt_sum)
        return melt_sum_vs_elevation

    def calc_melt_sum_catchment(self, swe3d):
        """
        Calculate the total SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        melt_sum = self.calc_melt_sum(swe_t)
        return melt_sum

    def calc_melt_sum_grid(self, swe3d):
        """
        Calculate the total SWE for each grid cell.
        """
        melt_sum_grid = xr.apply_ufunc(self.calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
        melt_sum_grid = xr.where(swe3d.isnull().all(dim='time'), np.nan, melt_sum_grid)
        return melt_sum_grid

    # Start and End Dates of SWE

    def calc_swe_dates(self,swe_t, smooth_window=5, threshold_frac=0.1):
        """
        Identify start and end of SWE season based on smoothed SWE values and thresholding.

        Parameters:
        -----------
        swe_t : array-like
            Time series of SWE values
        smooth_window : int
            Window size for smoothing (default: 5)
        threshold_frac : float
            Fraction of max SWE to use as threshold (default: 0.1)

        Returns:
        --------
        list
            [start_index, end_index] of the main snow period, or [np.nan, np.nan]
        """
        import numpy as np
        import pandas as pd

        if swe_t is None or len(swe_t) < 3 or np.all(np.isnan(swe_t)):
            return [np.nan, np.nan]

        swe_t = np.array(swe_t)
        swe_max = np.nanmax(swe_t)
        if swe_max <= 0:
            return [np.nan, np.nan]

        # Smooth the SWE to reduce noise
        swe_smooth = pd.Series(swe_t).rolling(window=smooth_window, center=True, min_periods=1).mean()

        # Threshold-based mask
        threshold = threshold_frac * swe_max
        mask = swe_smooth > threshold

        # Find continuous segments above threshold
        segments = []
        in_segment = False
        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                end = i - 1
                segments.append((start, end))
                in_segment = False
        if in_segment:
            segments.append((start, len(mask) - 1))

        if not segments:
            return [np.nan, np.nan]

        # Choose longest segment
        longest = max(segments, key=lambda x: x[1] - x[0])
        return [longest[0], longest[1]]

    # def calc_swe_dates(self, swe_t):
    #     """
    #     Calculate the start and end dates of SWE based on the 95th percentile of the maximum value.
        
    #     Parameters:
    #     -----------
    #     swe_t : array-like
    #         Time series of SWE values
        
    #     Returns:
    #     --------
    #     list
    #         [start_index, end_index] of the main snow period, or [np.nan, np.nan] if not identifiable
    #     """
    #     # Input validation
    #     if swe_t is None or len(swe_t) < 3:
    #         return [np.nan, np.nan]
        
    #     if np.all(np.isnan(swe_t)):
    #         return [np.nan, np.nan]
            
    #     # Calculate threshold based on 95th percentile of max
    #     swe_max = np.nanmax(swe_t)
    #     if swe_max <= 0:  # No real snow accumulation
    #         return [np.nan, np.nan]
            
    #     threshold = swe_max - np.nanquantile(swe_t, 0.95)
        
    #     # Get sign changes - these are potential crossing points
    #     above_threshold = swe_t > threshold
    #     sign_changes = np.where(np.diff(above_threshold))[0]
        
    #     # If no crossings, check if data starts above threshold
    #     if len(sign_changes) == 0:
    #         if above_threshold[0]:
    #             # Starts above threshold but never crosses down
    #             return [0, len(swe_t)-1]
    #         else:
    #             # Never crosses threshold
    #             return [np.nan, np.nan]
        
    #     # Handle case where data starts above threshold
    #     if above_threshold[0]:
    #         # First crossing is the end date
    #         if len(sign_changes) >= 1:
    #             return [0, sign_changes[0]]
    #         return [0, len(swe_t)-1]
        
    #     # Normal case - find the longest period above threshold
    #     if len(sign_changes) < 2:
    #         return [np.nan, np.nan]  # Need at least one complete period
        
    #     # Convert to start/end pairs
    #     periods = []
    #     for i in range(0, len(sign_changes), 2):
    #         if i+1 < len(sign_changes):
    #             start = sign_changes[i]
    #             end = sign_changes[i+1]
    #             periods.append((start, end, end-start))
        
    #     # If no complete periods found
    #     if not periods:
    #         return [np.nan, np.nan]
        
    #     # Find longest period
    #     longest_period = max(periods, key=lambda x: x[2])
        
    #     return [longest_period[0], longest_period[1]]
    # def calc_swe_dates(self, swe_t):
    #     """
    #     Calculate the start and end dates of SWE based on the 95th percentile.
    #     """
    #     q5 = np.nanmax(swe_t) - np.nanquantile(swe_t, 0.95)
    #     intercepts = np.where(np.diff(np.sign(swe_t - q5)))[0]
    #     if len(intercepts) == 1 and swe_t[0] > q5:
    #         return [0, intercepts[0]]
    #     if len(intercepts) > 2:
    #         distances = np.diff(intercepts)
    #         if len(distances) == 0:
    #             return [np.nan, np.nan]
    #         argmax = np.argmax(distances)
    #         intercepts = intercepts[argmax:argmax + 2]
    #     elif len(intercepts) < 2 or len(intercepts) != 2:
                
    #         return [np.nan, np.nan]
    #     return intercepts

    def calc_swe_start(self, swe_t):
        """
        Calculate the start date of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        intercepts = self.calc_swe_dates(swe_t)
        return intercepts[0]

    def calc_swe_end(self, swe_t):
        """
        Calculate the end date of SWE.
        """
        if np.all(np.isnan(swe_t)):
            return np.nan
        intercepts = self.calc_swe_dates(swe_t)
        return intercepts[1]

    def calc_swe_start_vs_elevation(self, swe3d):
        """
        Calculate the start date of SWE for different elevation bands.
        """
        starts = xr.apply_ufunc(self.calc_swe_start, swe3d, input_core_dims=[['time']], vectorize=True, output_dtypes=[np.float64])
        starts_vs_elevation = self.calc_var_vs_elevation(starts)
        return starts_vs_elevation

    def calc_swe_end_vs_elevation(self, swe3d):
        """
        Calculate the end date of SWE for different elevation bands.
        """
        ends = xr.apply_ufunc(self.calc_swe_end, swe3d, input_core_dims=[['time']], vectorize=True, output_dtypes=[np.float64])
        ends_vs_elevation = self.calc_var_vs_elevation(ends)
        return ends_vs_elevation

    def calc_swe_start_end_catchment(self, swe3d):
        """
        Calculate the start and end dates of SWE for the entire catchment.
        """
        swe_t = self.calc_sum2d(swe3d)
        intercepts = self.calc_swe_dates(swe_t)
        return intercepts

    def calc_swe_start_grid(self, swe3d):
        """
        Calculate the start date of SWE for each grid cell.
        """
        starts_grid = xr.apply_ufunc(self.calc_swe_start, swe3d, input_core_dims=[['time']], vectorize=True)
        return starts_grid

    def calc_swe_end_grid(self, swe3d):
        """
        Calculate the end date of SWE for each grid cell.
        """
        ends_grid = xr.apply_ufunc(self.calc_swe_end, swe3d, input_core_dims=[['time']], vectorize=True)
        return ends_grid

    # Kling-Gupta Efficiency (KGE)
    def calc_melt_kge(self, swe_obs_t, swe_sim_t):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates.
        """
        dif_obs = np.diff(swe_obs_t)
        dif_sim = np.diff(swe_sim_t)
        neg_dif_obs = np.where(dif_obs < 0, dif_obs * -1, 0)
        neg_dif_sim = np.where(dif_sim < 0, dif_sim * -1, 0)
        kge = he.kge_2009(neg_dif_obs, neg_dif_sim)
        return kge

    def calc_melt_kge_vs_elevation(self, swe_obs, swe_sim):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates for different elevation bands.
        """
        kges = xr.apply_ufunc(self.calc_melt_kge, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        kges_vs_elevation = self.calc_var_vs_elevation(kges)
        return kges_vs_elevation

    def calc_melt_kge_catchment(self, swe_obs, swe_sim):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        kge = self.calc_melt_kge(swe_obs_t, swe_sim_t)
        return kge

    def calc_melt_kge_grid(self, swe_obs, swe_sim):
        """
        Calculate the Kling-Gupta Efficiency (KGE) for melt rates for each grid cell.
        """
        kge_grid = xr.apply_ufunc(self.calc_melt_kge, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)
        return kge_grid

    def calc_melt_nse(self, swe_obs_t, swe_sim_t):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates.
        """
        dif_obs = np.diff(swe_obs_t)
        dif_sim = np.diff(swe_sim_t)
        neg_dif_obs = np.where(dif_obs < 0, dif_obs * -1, 0)
        neg_dif_sim = np.where(dif_sim < 0, dif_sim * -1, 0)
        nse = he.nse(neg_dif_obs, neg_dif_sim)
        return nse
    
    def calc_melt_nse_vs_elevation(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates for different elevation bands.
        """
        nses = xr.apply_ufunc(self.calc_melt_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        nses_vs_elevation = self.calc_var_vs_elevation(nses)
        return nses_vs_elevation

    def calc_melt_nse_catchment(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        nse = self.calc_melt_nse(swe_obs_t, swe_sim_t)
        return nse
    
    def calc_melt_nse_grid(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates for each grid cell.
        """
        nse_grid = xr.apply_ufunc(self.calc_melt_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)
        return nse_grid
    def calc_snowfall_nse(self, swe_obs_t, swe_sim_t):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for snowfall.
        """
        dif_obs = np.diff(swe_obs_t)
        dif_sim = np.diff(swe_sim_t)
        pos_dif_obs = np.where(dif_obs > 0, dif_obs, 0)
        pos_dif_sim = np.where(dif_sim > 0, dif_sim, 0)
        nse = he.nse(pos_dif_obs, pos_dif_sim)
        return nse
    def calc_snowfall_nse_vs_elevation(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe
        Efficiency (NSE) for snowfall for different elevation bands.
        """
        nses = xr.apply_ufunc(self.calc_snowfall_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        nses_vs_elevation = self.calc_var_vs_elevation(nses)
        return nses_vs_elevation
    def calc_snowfall_nse_catchment(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe
        Efficiency (NSE) for snowfall for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        nse = self.calc_snowfall_nse(swe_obs_t, swe_sim_t)
        return nse
    def calc_snowfall_nse_grid(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe
        Efficiency (NSE) for snowfall for each grid cell.
        """
        nse_grid = xr.apply_ufunc(self.calc_snowfall_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)
        return nse_grid

    # Nash-Sutcliffe Efficiency (NSE)
    def calc_swe_nse(self, swe_obs_t, swe_sim_t):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE.
        """
        nse = he.nse(swe_obs_t, swe_sim_t)
        return nse

    def calc_swe_nse_vs_elevation(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE for different elevation bands.
        """
        nses = xr.apply_ufunc(self.calc_swe_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64])
        nses_vs_elevation = self.calc_var_vs_elevation(nses)
        return nses_vs_elevation

    def calc_swe_nse_catchment(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE for the entire catchment.
        """
        swe_obs_t = self.calc_sum2d(swe_obs).to_pandas()
        swe_sim_t = self.calc_sum2d(swe_sim).to_pandas()
        nse = self.calc_swe_nse(swe_obs_t, swe_sim_t)
        return nse

    def calc_swe_nse_grid(self, swe_obs, swe_sim):
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE for each grid cell.
        """
        nse_grid = xr.apply_ufunc(self.calc_swe_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True)
        return nse_grid

    def calc_elev_bands(self, N_bands = 10):
        """
        Calculate elevation bands, making sure each bands has the same number of pixels.
        """
        dem = self.E.dem
        elev_flat = dem.values.flatten()
        elev_flat = pd.Series(elev_flat[~np.isnan(elev_flat)]).sort_values().reset_index(drop=True)
        quantiles = elev_flat.quantile(np.linspace(0, 1, N_bands + 1)).reset_index(drop=True)
        rounded_quantiles = np.round(quantiles, 0)
        self.elev_bands = rounded_quantiles
        return rounded_quantiles

    


    # Helper function to calculate variable vs elevation
    def calc_var_vs_elevation(self, var2d):
        """
        Calculate the mean of a variable in elevation bands defined by the calculated elevation bands.
        """
        dem = self.E.dem
        var2d_flat = var2d.values.flatten()
        dem_flat = dem.values.flatten()

        df = pd.DataFrame({'elevation': dem_flat, 'var': var2d_flat})
        df = df.dropna()
        # df['elevation_band'] = (df['elevation'] // 50) * 50
        # num_quantiles = 10  # You can adjust this number as needed
        # df['elevation_band'] = pd.qcut(df['elevation'], q=num_quantiles, labels=False)
        if not hasattr(self, 'elev_bands'):
            self.calc_elev_bands()
        elev_bands = self.elev_bands
        df['elevation_band'] = pd.cut(df['elevation'], bins=elev_bands, labels=False)

        var_by_band = (
            df.groupby('elevation_band')['var']
            .mean()
            .rename_axis('elevation_band')
            .reset_index(name='var')
            .set_index('elevation_band')
        )
        return var_by_band
        
    def calculate_spaef(self, swe_obs, swe_sim):
        """
        Calculate the SPAEF metric for assessing spatial patterns of SWE.
        Takes 3D data, turns it into 2D with melt_sum, and then calculates the SPAEF.

        Parameters:
        obs (xr.DataArray): Observed 3D data.
        sim (xr.DataArray): Simulated 3D data.

        Returns:
        float: SPAEF value.
        """
        # Ensure the input data are aligned
        # obs, sim = xr.align(obs, sim)

        obs = self.calc_melt_sum_grid(swe_obs)
        sim = self.calc_melt_sum_grid(swe_sim)

        # Flatten the data and remove NaNs
        obs_flat = obs.values.flatten()
        sim_flat = sim.values.flatten()
        mask = ~np.isnan(obs_flat) & ~np.isnan(sim_flat)
        obs_flat = obs_flat[mask]
        sim_flat = sim_flat[mask]

        # Calculate A: Pearson correlation coefficient
        A = pearsonr(obs_flat, sim_flat)[0]

        # Calculate B: Fraction of the coefficient of variation
        B = (variation(sim_flat) / variation(obs_flat))
        # Calculate C: Histogram intersection
        obs_hist, bin_edges = np.histogram(obs_flat, bins='auto', density=True)
        sim_hist, _ = np.histogram(sim_flat, bins=len(obs_hist), density=True)
        C = np.sum(np.minimum(obs_hist, sim_hist)) / np.sum(obs_hist)
        # C basically gives the total overlap of the two histogram. So if you then to C-1 you get the error

        # Calculate SPAEF
        SPAEF = 1 - np.sqrt((A - 1)**2 + (B - 1)**2 + (C - 1)**2)

        return SPAEF

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
        
    #Functions to calculate Q signatures
    def calc_snowmeltseason_cv(self,Q, months = [4,5,6,7]):
        Q = Q.loc[Q.index.month.isin(months)]
        cv = Q.std()/Q.mean()
        return cv
    def calc_snowmeltseason_sum(self,Q, months = [4,5,6,7]):
        Q = Q.loc[Q.index.month.isin(months)]
        sum = Q.sum()
        return sum
    def calc_Q_skew(self,Q):
        skew = Q.skew()
        return skew
    def calc_Q_var(self,Q):
        var = Q.var()
        return var
    def calc_Q_cov(self,Q):
        cov = Q.std()/Q.mean()
        return cov
    def calc_t_qmax(self,Q):
        t_qmax = Q.idxmax()
        #convert to days since October 1st
        t_qmax = (t_qmax - pd.Timestamp(f"{t_qmax.year-1}-10-01")).days
        return t_qmax
    def calc_half_flow_date(self,Q):
        """
        Calculate the half flow date (HFD) for a given time series of streamflow (Q) and corresponding dates (t).
        The HFD is the date when half of the annual flow has passed.
        """
        # Ensure Q and t are numpy arrays
        Q_half_sum = 0.5 * np.sum(Q)
        Q_cumsum = np.cumsum(Q)
        HFD_aux = np.where(Q_cumsum > Q_half_sum)[0]
        if len(HFD_aux) > 0:
            HFD = HFD_aux[0]
        else:
            HFD = np.nan
        return HFD
    def calc_half_flow_interval(self,Q):
        """Calculates time span between the date on which the cumulative discharge 
        %   since start of water year (default: October) reaches (here: exceeds) a 
        %   quarter of the annual discharge and the date on which the cumulative 
        %   discharge since start of water year (default: October) reaches three 
        %   quarters of the annual discharge."""
        Q_quarter = 0.25 * np.sum(Q)
        Q_three_quarters = 0.75 * np.sum(Q)
        Q_cumsum = np.cumsum(Q)
        HFI_aux1  = np.where(Q_cumsum > Q_quarter)[0]
        HFI_aux2  = np.where(Q_cumsum > Q_three_quarters)[0]
        HFI = HFI_aux2[0] - HFI_aux1[0]
        return HFI
    def calc_high_flow_freq(self,Q, pct =0.9):
        """Calculates the frequency of high flow events, defined as the number of days with streamflow exceeding a given threshold."""
        Qthreshold = pct * Q.max()
        high_flow_freq = np.sum(Q > Qthreshold)/len(Q)
        return high_flow_freq
    def calc_peak_distribution(self,Q,slope_range=(0.1, 0.5), fit_log_space=False):
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(Q)
        Q_peak = Q.iloc[peaks]

        # Sort the peaks and calculate their ranks
        Q_peak_sorted = np.sort(Q_peak)[-1::-1]
        # Q_peak_ranked = np.arange(1, len(Q_peak_sorted) + 1)
        Q90 = np.quantile(Q,0.9)
        Q50 = np.quantile(Q,0.5)
        peaks_dist = (Q90-Q50)/(0.4)
        return peaks_dist
    def calc_flashiness_index(self,Q):
        from hydrosignatures import flashiness_index as fi
        flashiness = fi(Q).item()
        return flashiness
    def calc_Q_inflection(self,Q):
        """Horner2020 use a 30day smoothing on Q and then take the maximum
          of the second derivative of Q_cumsum to find the inflection point.
          They show it to be related to the timing of SWEmax """
        Q_cumsum = np.cumsum(Q).squeeze().rolling(window=30).mean()

        # Calculate the first and second derivatives
        first_derivative = np.gradient(Q_cumsum)
        second_derivative = np.gradient(first_derivative)
        first_inflection_point = np.nanargmax(second_derivative)
        second_inflection_point = np.nanargmin(second_derivative)
        return first_inflection_point
    def peakfilter_mask(self,array):
        diff = np.diff(array,prepend=0)
        posdif = (diff>0) * diff
        mask1 = posdif> 2*np.std(posdif)
        mask1_shifted = np.roll(mask1, -1)
        mask1_shifted2 = np.roll(mask1, -2)
        mask1_shifted3 = np.roll(mask1, -3)
        # Create a new mask that includes the day after each storm peak
        mask2 = mask1 | mask1_shifted | mask1_shifted2 | mask1_shifted3
        mask3 = array > np.quantile(array, 0.95)
        mask = mask2 | mask3
        return mask
    def peakfilter(self,Q_array):
        QQ = Q_array.squeeze().copy()
        mask = self.peakfilter_mask(QQ)
        filtered = np.where(mask, QQ, np.nan)
        QQ[mask] = np.nan
        return QQ
    def baseflow_filter(self,Q_array):
        from hydrosignatures.baseflow import baseflow
        QQ = Q_array.squeeze().copy()
        bf = baseflow(QQ)
        if np.all(bf == QQ):
            print("Something is wrong, maybe try Q_array[:,0]")
        return bf
    
    def calc_Q_amplitude(self,Q_array):
        # Take the 30 day rolling mean of the streamflow and calculate the amplitude
        Q = Q_array.squeeze().copy()
        Q_smoothed = Q.rolling(window=30).mean()
        Q_amplitude = Q_smoothed.max() - Q_smoothed.min()
        return Q_amplitude
    def calc_Qstart(self,Q_array):
        Q = Q_array.squeeze().copy()
        window = 30
        Q_smoothed = Q.rolling(window=window).mean()
        pct  = np.nanpercentile(Q_smoothed,5)
        intercepts = np.where(np.diff(np.sign(Q_smoothed - pct)))[0]
        intercepts = intercepts[intercepts>window]
        if len(intercepts) == 0:
            return np.nan
        elif len(intercepts) ==2:
            Qstart = intercepts[1]
        elif len(intercepts)==1:
            print("Only one intercept wiht Q05")
            if Q_smoothed[intercepts[0]]>Q_smoothed[intercepts[0]-1]:
                Qstart = intercepts[0]
            else:
                Qstart = np.nan
        elif len(intercepts) >2:
            distances = np.diff(intercepts)
            argmax = np.argmax(distances)
            intercept = intercepts[argmax]
            Qstart = intercept
        return Qstart
    def calc_t_Qrise(self,Q_array):
        Qstart = self.calc_Qstart(Q_array)
        Q = Q_array.squeeze().copy()
        Q_smoothed = Q.rolling(window=30).mean()
        t_Qmax = np.nanargmax(Q_smoothed)
        t_Qrise = t_Qmax - Qstart
        return t_Qrise

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



    def plot_QSWE_signatures(self):
        print('Plotting Q and SWE signatures')
        for year in range(self.START_YEAR, self.END_YEAR+1):
            corr_mat = self.metrics[year].corr().stack().reset_index(name='correlation')
            swemask = corr_mat.level_0.str.contains('SWE')
            notswemask =~corr_mat.level_1.str.contains('SWE')
            corr_mat = corr_mat[swemask & notswemask]
            corr_mat['correlation'] **=2
            

            g = sns.relplot( data=corr_mat,
                x="level_1", y="level_0", hue="correlation", size="correlation",
                palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
                height=10, sizes=(250, 250), size_norm=(-.2, .8),
            )

            # Tweak the figure to finalize
            g.set(xlabel="", ylabel="", aspect="equal")
            g.despine(left=True, bottom=True)
            g.ax.margins(0.02,0.1)
            for label in g.ax.get_xticklabels():
                label.set_rotation(80)
            plt.title(f'R2 between SWE and Q signatures \n {year}')
            plt.savefig(join(self.FIGDIR, f'Q_SWE_signatures_{year}.png'),
                        dpi = 300, bbox_inches = 'tight')

        #One figure with all yeras together
        metrics_concat = pd.concat([self.metrics[year] for year in range(self.START_YEAR, self.END_YEAR+1)],axis=0)
        corr_mat = metrics_concat.corr().stack().reset_index(name='correlation')
        swemask = corr_mat.level_0.str.contains('SWE')
        notswemask =~corr_mat.level_1.str.contains('SWE')
        corr_mat = corr_mat[swemask & notswemask]
        corr_mat['correlation'] **=2

        g = sns.relplot(data=corr_mat,
            x="level_1", y="level_0", hue="correlation", size="correlation",
            palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
            height=10, sizes=(250, 250), size_norm=(-.2, .8),
        )
        g.set(xlabel="", ylabel="", aspect="equal")
        g.despine(left=True, bottom=True)
        g.ax.margins(0.02,0.1)
        for label in g.ax.get_xticklabels():
            label.set_rotation(80)
        plt.title(f'R2 between SWE and Q signatures \n {self.START_YEAR}-{self.END_YEAR}')
        plt.savefig(join(self.FIGDIR, f'Q_SWE_signatures_{self.START_YEAR}_{self.END_YEAR}.png'),
                        dpi = 300, bbox_inches = 'tight')

    def compute_QSWE_corr(self):
        corr_list = []
        for year in range(self.START_YEAR, self.END_YEAR+1):
            corr_mat = self.metrics[year].corr(method = 'spearman').stack().reset_index(name='correlation')
            swemask = corr_mat.level_0.str.contains('SWE') | corr_mat.level_0.str.contains('melt_sum')
            notswemask = ~corr_mat.level_1.str.contains('SWE') & ~corr_mat.level_1.str.contains('melt_sum')
            corr_mat = corr_mat[swemask & notswemask]
            corr_mat['Year'] = year
            # corr_mat['correlation'] **=2
            corr_list.append(corr_mat)
        QSWE_corr = pd.concat(corr_list,axis=0)
        #filter QSWE_corr based on a list of Q metric names
        q_metrics = self.q_metrics_to_use
        mask = QSWE_corr['level_1'].isin(q_metrics)
        QSWE_corr = QSWE_corr[mask]

        swe_metrics = self.swe_metrics_to_use
        mask = QSWE_corr['level_0'].isin(swe_metrics)
        QSWE_corr = QSWE_corr[mask]
        #Convert negative corrleations to positive when negative is good
        for qm in q_metrics:
            if 'KGE' in qm or 'NSE' in qm:
                QSWE_corr.loc[QSWE_corr['level_1']==qm,'correlation'] *= -1
        for sm in np.unique(QSWE_corr['level_0']):
            if 'KGE' in sm or 'NSE' in sm:
                QSWE_corr.loc[QSWE_corr['level_0']==sm,'correlation'] *= -1
        self.QSWE_corr = QSWE_corr
        #save to file 
        QSWE_corr_file = join(self.LOA_DIR,'QSWE_corr.csv')
        QSWE_corr.to_csv(QSWE_corr_file)
        return QSWE_corr
    def plot_QSWE_signatures_stripplot(self):
        if not hasattr(self,'QSWE_corr'):
            self.compute_QSWE_corr()
        QSWE_corr = self.QSWE_corr
        q_metrics = self.q_metrics_to_use
        f1,ax1 = plt.subplots(figsize = (5,0.7*len(q_metrics)))
        sns.stripplot(QSWE_corr, hue = "level_0", y = 'level_1', x = 'correlation', orient = 'h',
                      dodge = True, palette = self.swe_palette, ax = ax1)
        ax1.set_xlabel('Spearman rank correlation \n (Positive = good)')
        ax1.set_ylabel('Q metric')
        ax1.grid()
        ax1.axvline(0, color = 'black',  linestyle = '--')
        ax1.set_xlim(-1,1)
        ax1.legend(loc = (1.05,0.5))
        ax1.set_title(f"{self.EXP_ID_translation[self.ORIG_ID]}")
        plt.savefig(join(self.FIGDIR, 'QSWE_signatures_stripplot_Q.png'),
                        dpi = 300, bbox_inches = 'tight')

        f1,ax1 = plt.subplots(figsize = (5,0.7*len(q_metrics)))
        sns.stripplot(QSWE_corr, hue = "level_1", y = 'level_0', x = 'correlation', orient = 'h',
                      dodge = True, palette = self.Q_palette, ax = ax1)
        ax1.set_xlabel('Spearman rank correlation \n (Positive = good)')
        ax1.set_ylabel('SWE metric')
        ax1.grid()
        ax1.axvline(0, color = 'black',  linestyle = '--')
        ax1.set_xlim(-1,1)
        ax1.legend(loc = (1.05,0.5))
        ax1.set_title(f"{self.EXP_ID_translation[self.ORIG_ID]}")

        plt.savefig(join(self.FIGDIR, 'QSWE_signatures_stripplot_SWE.png'),
                        dpi = 300, bbox_inches = 'tight')
    def plot_best_QSWE_corrs(self):
        if not hasattr(self,'QSWE_corr'):
            self.compute_QSWE_corr()
        QSWE_corr = self.QSWE_corr
        median_correlations = QSWE_corr.groupby(['level_0', 'level_1'])['correlation'].mean().reset_index().sort_values('correlation', ascending=False)
        topcombos = [[row['level_0'], row['level_1']] for i, row in median_correlations.head(10).iterrows()]
        # worstcombos = [[row['level_0'], row['level_1']] for i, row in median_correlations.tail(10).iterrows()]
        # topcombos = topcombos + worstcombos
        # topcombos.append(['SWE_max_elev_ME', 'NSE'])

        # Print topcombos to debug
        print("Top Combos:", topcombos)

        # Convert topcombos to a list of tuples for easier filtering
        topcombos_tuples = [tuple(combo) for combo in topcombos]
        self.topcombos_tuples = topcombos_tuples

        QSWE_corr_top = QSWE_corr[QSWE_corr.apply(lambda x: (x['level_0'], x['level_1']) in topcombos_tuples, axis=1)]
        QSWE_corr_top['rank'] = np.nan
        for r, combo in enumerate(topcombos):
            print(r, combo)
            QSWE_corr_top.loc[(QSWE_corr_top['level_0'] == combo[0]) & (QSWE_corr_top['level_1'] == combo[1]), 'rank'] = r
        
        q_metrics = self.q_metrics_to_use

        f1, ax1 = plt.subplots(figsize=(5, 0.3 * len(q_metrics)))
        sns.stripplot(data=QSWE_corr_top, ax = ax1,
                      hue="level_0", 
                      y='rank',
                        x='correlation',
                          orient='h', 
                          dodge=False, palette=self.swe_palette)
        ax1.set_yticklabels([f"{i+1}. {tc[1]}" for i,tc in enumerate(topcombos)])
        ax1.grid()
        ax1.axvline(0, color = 'black',  linestyle = '--')
        ax1.set_xlim(-1,1)
        ax1.legend(loc = (1.05,0.5))
        ax1.set_xlabel('Spearman rank correlation \n (Positive = good)')
        ax1.set_ylabel('Rank + Q metric')
        ax1.set_title(f"Top 10 Q-SWE correlations \n Each point is one year between 2001-2022  ")
        f1.suptitle(f"{self.EXP_ID_translation[self.ORIG_ID]}", y = 1.1, fontsize = 16)
        plt.savefig(join(self.FIGDIR, 'best_QSWE_corrs.png'),
        dpi = 300, bbox_inches = 'tight')


    def dotty_plot(self,metric1,metric2):
        # Create a figure and axis


        color = sns.color_palette("viridis", n_colors = self.END_YEAR-self.START_YEAR+1)
        fig, ax = plt.subplots()
        for i,year in enumerate([2003]):#range(self.START_YEAR, self.END_YEAR+1)):
            df1 = self.metrics[year][metric1]
            df2 = self.metrics[year][metric2]
            sns.scatterplot(x = df1,y= df2, s = 5,
                            alpha=0.5, color = 'black', #color[i], 
                            ax = ax, zorder = 1)
            # sns.kdeplot(data = self.metrics[year],
            #             x = metric1,
            #             y = metric2,
            #             fill = True,
            #             cmap = 'rocket',
            #             # color = 'blue',#color[i],
            #             ax = ax,
            #             alpha = 0.5,thresh= 0.0,
            #             label = year)
            # sns.kdeplot(data = self.metrics[year],
            #             x = metric1,
            #             y = metric2,
            #             fill = False,
            #             color = color[i],
            #             ax = ax,
            #             alpha = 0.5)
            # sns.regplot(data = self.metrics[year],
            #             x = metric1,
            #             y = metric2,
            #             color = color[i],
            #             ax = ax,
            #             scatter = False, label = year) 
            # sns.histplot(data = self.metrics[year], x=metric1, y=metric2, 
            #              bins=50, pthresh=0, cmap="mako",alpha = 0.5)
            plt.ylabel(metric2)
            plt.xlabel(metric1)
        plt.grid(alpha = 0.5)
        plt.xlim(-1,1.1)
        plt.ylim(0,70)

        plt.legend()
        plt.savefig(join(self.FIGDIR, f'dotty_plot_{metric1}_{metric2}.png'))
    
    def QSWEplot(self,Qmetric, pct = 0.01):
        """2 rows of plots  
        1st row: Q
            - Qobs
            - All Q in alpha = 0.001
            - Q that score high on metric X 
        2nd row:SWE 
            - SWEobs
            - All SWE in alpha = 0.001
            - SWE that score high on metric X
        """
        from matplotlib.lines import Line2D

        # Qmetric = 'Qmean_meltseason_ME'
        # Qmetric = 'KGE'
        # Qmetric = 't_hfd_ME'
        # Qmetric = 'bfi_ME'
        # pct = 0.01

        # years = np.arange(self.START_YEAR, self.END_YEAR + 1)
        years = np.arange(self.START_YEAR, self.START_YEAR+10)
        Naxes = len(years)
        f1, axes = plt.subplots(2, Naxes, figsize=(3 * Naxes, 8))

        for year in years:
            Qtime = slice(f"{year}-04-01", f"{year}-07-30")
            # Q
            ax = axes[1, year - self.START_YEAR]
            qobs = self.Qobs.loc[Qtime]
            qobs.plot(ax=ax, color='black', zorder=1e6, legend=False)
            qall = self.Q.loc[Qtime]
            ax.fill_between(qall.index, qall.min(axis=1), qall.max(axis=1), color='grey', alpha=0.2)
            # qall.iloc[:,:int(0.5*len(qall))].plot(ax=ax, color='grey', alpha=0.05, legend=False)
            m = self.metrics[year]
            # Take the pct% best score on metric X
            if Qmetric in ['KGE','NSE','R2','KGE_meltseason']:
                indices = m[Qmetric].nlargest(int(np.ceil(pct * len(m)))).index
            else:
                indices = m[Qmetric].nsmallest(int(np.ceil(pct * len(m)))).index
            qbest = qall.loc[:, indices]
            qbest.plot(ax=ax, color='tab:blue', alpha=0.3, legend=False)

            ax.set_title(f"Q for {year}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Discharge")
            ax.grid(True)

            # SWE
            axswe = axes[0, year - self.START_YEAR]
            swetime = slice(f"{year-1}-10-01", f"{year}-06-30")
            sweobs = self.SWEobs.mean(dim=['lat', 'lon']).sel(time=swetime).to_pandas()
            sweobs.plot(ax=axswe, color='black', zorder=1e6, legend=False)
            sweall = self.SWE[year].sel(time=swetime).mean(dim=['lat', 'lon']).to_pandas().drop(columns='spatial_ref')
            maxes = sweall.max(axis = 1)
            minusses = sweall.min(axis = 1)
            axswe.fill_between(sweall.index, minusses, maxes, color = 'grey', alpha = 0.2)
            # sweall.iloc[:,:int(0.5*len(sweobs))].plot(ax=axswe, color='grey', alpha=0.05, legend=False)
            #select indices from sweall, but only if they exist 
            adjusted_indices = [i for i in indices if i in sweall.columns]
            swebest = sweall.loc[:, adjusted_indices]
            swebest.plot(ax=axswe, color='tab:blue', alpha=0.3, legend=False)

            axswe.set_title(f"SWE for {year}")
            axswe.set_xlabel("Date")
            axswe.set_ylabel("SWE")
            axswe.grid(True)

        # Create custom legend
        custom_lines = [Line2D([0], [0], color='black', lw=2),
                        Line2D([0], [0], color='grey', alpha=0.1, lw=2),
                        Line2D([0], [0], color='tab:blue', alpha=0.3, lw=2)]
        f1.legend(custom_lines, ['Qobs','All plots', 'Selection'], loc='upper right')
        f1.suptitle(f"Q and SWE for the {pct}% best {Qmetric} runs")
        plt.tight_layout()
        plt.savefig(join(self.FIGDIR, f'QSWEplot_{Qmetric}.png'))
        plt.show()

    def compute_Qsig_corr(self):
        Qsigs =pd.concat([self.Q_signatures[year]['sim'] for year in range(self.START_YEAR, self.END_YEAR+1)],axis=0)
        # q_metrics = ['Qmean', 'Qmax','t_hfd',
        #                 't_Qinflection','Qmean_meltseason','Qcv_meltseason',
        #                 'BaseFlow_sum',
        #                 'bfi','Qcv','Qamp','t_Qstart','t_Qrise']
        q_sigs_to_use = [qm[:-3] for qm in self.q_metrics_to_use if 'ME' in qm]
        # Qsigs = Qsigs[q_metrics]
        Qsigcorr = Qsigs.corr().abs()

        sns.heatmap(Qsigcorr, cmap = 'coolwarm')
        threshold = 0.7
        df = pd.DataFrame(columns = ['corr','s1','s2'])
        for i in range(len(Qsigcorr.columns)):
            for j in range(i):
                corrr = np.round(abs(Qsigcorr.iloc[i, j]),2)
                if  corrr> threshold:
                    df = pd.concat([df,pd.DataFrame({'corr':corrr,'s1':Qsigcorr.columns[i],'s2':Qsigcorr.columns[j]},index = [0])],axis = 0)
        df.sort_values('corr',ascending = False, inplace = True)
        print("Highly correlated signatures")
        print(df)
        print()

        for i in range(df.shape[0]):
            if df.iloc[i,1] in q_sigs_to_use and df.iloc[i,2] in q_sigs_to_use:
                print(f"{df.iloc[i,1]} and {df.iloc[i,2]} in metrics to use and correlated {df.iloc[i,0]}")

    def compute_swesig_elev_corr(self):
        metrics = list(self.swe_signatures[2010]['sim'].keys())        
        bands = self.swe_signatures[self.START_YEAR]['sim'][metrics[0]].index 
        swemetrics_perband = {}
        for b in bands:
            # base_df = pd.DataFrame(columns = metrics)
            df_list = []
            for year in range(self.START_YEAR, self.END_YEAR+1):
                # df = self.swe_signatures[year]['sim']
                base_df = pd.DataFrame(columns = metrics)
                for m in metrics:
                    base_df[m] = self.swe_signatures[year]['sim'][m].loc[b,:]
                df_list.append(base_df) 
            total_df = pd.concat(df_list,axis=0)
            swemetrics_perband[b] = total_df
        
        # for b in bands[::3]:
        #     swemetrics = swemetrics_perband[b]
        #     swemetrics_corr = swemetrics.corr().abs()
        #     plt.figure()
        #     sns.heatmap(swemetrics_corr, cmap = 'coolwarm')
        #     plt.title(f"Correlation between SWE metrics for band {b}")
        
        #concat all swemetrics_perband into one df
        swemetrics_concat = pd.concat([swemetrics_perband[b] for b in bands],axis=0)
        swemetrics_corr = swemetrics_concat.corr().abs()
        plt.figure()
        sns.heatmap(swemetrics_corr, cmap = 'coolwarm')
        plt.title(f"Correlation between SWE metrics for all bands")

    def compute_metric_corr(self,metrics, metrics_to_use):
        mselection = pd.concat([metrics[year] for year in range(self.START_YEAR, self.END_YEAR+1)],axis=0)
        # mselection = mselection[self.q_metrics_to_use]
        corr = mselection.corr().abs()

        plt.figure(figsize = (15,15))
        sns.heatmap(corr, cmap = 'coolwarm')
        threshold = 0.7
        to_remove = set()
        pairs = {}
        df = pd.DataFrame(columns = ['corr','m1','m2'])
        # l = []
        for i in range(len(corr.columns)):
            for j in range(i):
                corrr = np.round(abs(corr.iloc[i, j]),2)
                if  corrr> threshold:
                    df = pd.concat([df,pd.DataFrame({'corr':corrr,'m1':corr.columns[i],'m2':corr.columns[j]},index = [0])],axis = 0)
        df.sort_values('corr',ascending = False, inplace = True)
        print("Highly correlated metrics")
        print(df)

        for i in range(df.shape[0]):
            if df.iloc[i,1] in metrics_to_use and df.iloc[i,2] in self.q_metrics_to_use:
                print(f"{df.iloc[i,1]} is highly correlated with {df.iloc[i,2]}")
                #remove df.iloc[i,2] from the list of metrics to use
                to_remove.add(df.iloc[i,2])
                # pairs[df.iloc[i,1]] = df.iloc[i,2]
        new_metrics_to_use = [metric for metric in metrics_to_use if metric not in to_remove]

        return new_metrics_to_use

    def calc_corr_combos(self,Nmax=4):
        import itertools
        def generate_combinations(selection, N):
            return ["+".join(combo) for combo in itertools.combinations(selection, N)]

        # Example usage
        q_combos = []
        swe_combos = []
        for N in range(2, Nmax + 1):
            q_combos += generate_combinations(q_selection, N)
            swe_combos += generate_combinations(swe_selection, N)

        q_combos = [q_combo for q_combo in q_combos if np.all(['ME' in qm for qm in q_combo.split('+')])]

        m_rank_dic = {}
        for year in range(self.START_YEAR, self.END_YEAR+1):
            m = self.metrics[year]
            m_rank = m.rank(ascending = True)
            for col in m_rank.columns:
                # if col in ['KGE','NSE','R2']:
                if 'KGE' in col or 'NSE' in col or 'R2' in col:
                    m_rank[col] = m_rank[col].rank(ascending = False)
            for cc in q_combos+swe_combos:
                metrics = cc.split('+')
                rankie = sum(m_rank[m] for m in metrics).rank(ascending=True)
                m_rank[cc] = rankie
            m_rank_dic[year] = m_rank
        self.m_rank_dic = m_rank_dic

        corr_list = []
        for year in range(self.START_YEAR, self.END_YEAR+1):
            corr_mat = m_rank_dic[year].corr(method = 'spearman').stack().reset_index(name='correlation')
            swemask = corr_mat.level_0.str.contains('SWE') | corr_mat.level_0.str.contains('melt_sum')
            notswemask = ~corr_mat.level_1.str.contains('SWE') & ~corr_mat.level_1.str.contains('melt_sum')
            corr_mat = corr_mat[swemask & notswemask]
            corr_mat['Year'] = year
            # corr_mat['correlation'] **=2
            corr_list.append(corr_mat)
        self.QSWE_corr_combos = pd.concat(corr_list,axis=0)

        self.bestcorr_combos = self.QSWE_corr_combos.groupby(['level_0', 'level_1'])['correlation'].mean().reset_index().sort_values('correlation', ascending=False)
        print(self.bestcorr_combos)

    def calculate_LOA_metrics(self, Q_chosen):
        LOA_metrics = {}
        for Qm in Q_chosen:
            if ('ME' in Qm) or ('APE' in Qm):
                metric = '_'.join(Qm.split('_')[:-1])
                Qsig_obs = [self.Q_signatures[year]['obs'][metric] for year in range(self.START_YEAR, self.END_YEAR + 1)]
                LOA_metrics[Qm] = 0.15 * np.nanstd(Qsig_obs)
            else:
                LOA_metrics[Qm] = 0.02
        # self.LOA_metrics = LOA_metrics
        return LOA_metrics

    def perform_LOA_checks(self, Q_chosen, LOA_metrics, flat_percentage = None):
        LOA_checks = {}
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            runids = self.metrics[self.START_YEAR].index

            if flat_percentage == None:
                LOA_checks[year] = pd.DataFrame(columns=Q_chosen, index=runids)

                for Qm in Q_chosen:
                    if ('ME' in Qm) or ('APE' in Qm):
                        metric = '_'.join(Qm.split('_')[:-1])
                        Qsig_obs = self.Q_signatures[year]['obs'][metric]
                        upper = Qsig_obs + LOA_metrics[Qm]
                        lower = Qsig_obs - LOA_metrics[Qm]
                        for runid in runids:
                            Qsig_sim = self.Q_signatures[year]['sim'][metric].loc[runid]
                            check = (Qsig_sim > lower) & (Qsig_sim < upper)
                            LOA_checks[year].loc[runid, Qm] = check.item()
                    else:
                        Qm_max = self.metrics[year][Qm].max()
                        checks = self.metrics[year][Qm] > Qm_max - LOA_metrics[Qm]
                        LOA_checks[year][Qm] = checks

                LOA_checks[year]['all'] = LOA_checks[year].all(axis=1)
            else:
                LOA_checks[year] = pd.DataFrame(columns=['all'], index=runids)
                # if 'NSE' in Q_chosen[0] or 'KGE' in Q_chosen[0]:
                #     ascending = False
                # else:
                #     ascending = True
                # top_ids = metrics.sort_values(ascending=ascending).index
                # checks = 
                # # checks = self.metrics[year][Q_chosen[0]].loc[top_ids] > self.metrics[year][Q_chosen[0]].max() - LOA_metrics[Q_chosen[0]]
                # # LOA_checks[year]['all'].iloc[:int(flat_percentage*0.01 * len(top_ids))] = True
                # # LOA_checks[year]['all'].iloc[int(flat_percentage*0.01 * len(top_ids)):] = False
                # # LOA_checks[year]['all'] = LOA_checks[year]['all'].astype(bool)
                # LOA_checks[year]['all'] = checks
                metrics = self.metrics[year][Q_chosen[0]]
                if 'NSE' in Q_chosen[0] or 'KGE' in Q_chosen[0]:
                    ascending = False
                else:
                    ascending = True
                
                # Sort metrics based on ascending/descending order
                sorted_metrics = metrics.sort_values(ascending=ascending)
                
                n_runs_to_include = min(int(flat_percentage * 0.01 * len(sorted_metrics)),
                                         int(flat_percentage * 0.01 * self.runid_limit))
                    
                # Create boolean mask, True for top n_runs, False for the rest
                LOA_checks[year]['all'] = False
                top_ids = sorted_metrics.index[:n_runs_to_include]
                LOA_checks[year].loc[top_ids, 'all'] = True


        # self.LOA_checks = LOA_checks
        return LOA_checks

    def gather_SWE_results(self, LOA_checks, Q_chosen, SWE_chosen, Lower_Benchmark=None):
        SWE_results_dic = {}
        SWE_results_list = []

        for year in range(self.START_YEAR, self.END_YEAR + 1):
            Q_posterior = LOA_checks[year]['all'][LOA_checks[year]['all']].index.tolist()
            # if len(Q_posterior) > 0:
            print(f"Year {year} has {len(Q_posterior)} good runs for metric {Q_chosen}")

            SWEm_all = self.metrics[year][SWE_chosen]
            SWEm_posterior = SWEm_all.loc[Q_posterior]
            # SWEm_UB = pd.Series({SWEm_all.idxmin(): SWEm_all.loc[SWEm_all.idxmin()]})
            if 'KGE' in SWE_chosen or 'NSE' in SWE_chosen or 'SPAEF' in SWE_chosen:
                ascending = False
            else:
                ascending = True
            SWEm_UB = SWEm_all.sort_values(ascending=ascending).head(int(len(SWEm_all)/100))
            # SWEm_LB = SWEm_all.sample(n=50, replace=True)
            if Lower_Benchmark==None:
                SWEm_LB = SWEm_all.sample(n=50, replace=True)
            else:
                SWEm_LB = Lower_Benchmark.metrics[year][SWE_chosen]

            SWE_results_dic[year] = {
                'Prior': SWEm_all,
                'Posterior': SWEm_posterior,
                'UB': SWEm_UB,
                'LB': SWEm_LB
            }

            # Convert to melted dataframe
            for cat, data in SWE_results_dic[year].items():
                m = data.to_frame(name=SWE_chosen)
                m['category'] = cat
                m['year'] = year
                m['Q_metric'] = Q_chosen
                SWE_results_list.append(m)

        # self.SWE_results_dic = SWE_results_dic
        SWE_results = pd.concat(SWE_results_list, axis=0)#.reset_index()

        # sns.boxenplot(data = SWE_results, x = SWE_chosen, y = 'category', hue = 'year', dodge = True)
        return SWE_results

        
    def make_Qsig_obs(self):
        """
        Construct a DataFrame of observed Q-signatures and climate metrics
        for each water year from loa.START_YEAR to loa.END_YEAR.
        
        Parameters
        ----------
        loa : object
            An object with attributes:
            - START_YEAR, END_YEAR (ints)
            - Q_signatures: dict of dicts, keyed by year then 'obs'
            - meteo_base: xarray Dataset with 'pr' and 'tas'
            - SWEobs: xarray DataArray of observed SWE
            - calc_melt_sum_catchment: method taking a sliced SWEobs
        
        Returns
        -------
        Qsig_obs : pd.DataFrame
            Indexed by year, with one column per Q-signature plus:
            'Annual Precip', 'Nov–May Precip',
            'Annual Temp', 'Winter Temp', 'Snowfall frac'
        """
        years = list(range(self.START_YEAR, self.END_YEAR + 1))
        
        # 1) Q-signatures
        Qm_list = self.Q_signatures[self.START_YEAR]['obs'].index.tolist()
        Qsig_obs = pd.DataFrame(index=years, columns=Qm_list)
        for Qm in Qm_list:
            Qsig_obs[Qm] = [
                self.Q_signatures[y]['obs'][Qm] for y in years
            ]
        
        # 2) Annual precip (Oct–Sep)
        pr = self.meteo_base['pr'].mean(dim=['lat','lon'])
        pr_yr = (
            pr.resample(time='AS-OCT').sum().to_pandas()
            .loc[f"{self.START_YEAR-1}":f"{self.END_YEAR}"]
        )
        pr_yr.index = pr_yr.index.year + 1
        Qsig_obs['Annual Precip'] = pr_yr.loc[years].values
        
        # 3) Nov–May precip
        pr_mon = pr.resample(time='M').sum().to_pandas()
        sel = pr_mon[pr_mon.index.month.isin([11,12,1,2,3,4,5])]
        pr_nm = sel.resample('AS-OCT').sum()
        pr_nm.index = pr_nm.index.year + 1
        Qsig_obs['Nov–May Precip'] = pr_nm.loc[years].values
        
        # 4) Annual temperature
        tas = self.meteo_base['tas'].mean(dim=['lat','lon'])
        tas_yr = (
            tas.resample(time='AS-OCT').mean().to_pandas()
            .loc[f"{self.START_YEAR-1}":f"{self.END_YEAR}"]
        )
        tas_yr.index = tas_yr.index.year + 1
        Qsig_obs['Annual Temp'] = tas_yr.loc[years].values
        
        # 5) Winter temperature (Nov–May)
        tas_mon = tas.resample(time='M').mean().to_pandas()
        tw = tas_mon[tas_mon.index.month.isin([11,12,1,2,3,4,5])]
        tw_yr = tw.resample('AS-OCT').mean()
        tw_yr.index = tw_yr.index.year + 1
        Qsig_obs['Winter Temp'] = tw_yr.loc[years].values
        
        # 6) Snowfall fraction (Oct–Sep)
        sf_frac = []
        for y in years:
            start, end = f"{y-1}-10-01", f"{y}-09-30"
            snow_sum = self.calc_melt_sum_catchment(
                self.SWEobs.sel(time=slice(start, end))
            )
            pr_sum = (
                self.meteo_base['pr']
                .sel(time=slice(start, end))
                .mean(dim=['lat','lon'])
                .sum()
                .item()
            )
            sf_frac.append(snow_sum / pr_sum if pr_sum else 0)
        Qsig_obs['Snowfall frac'] = sf_frac
        self.Qsig_obs = Qsig_obs
        
        return Qsig_obs


#%%
if __name__ == '__main__':
    config_dir = "/home/pwiersma/scratch/Data/ewatercycle/experiments/config_files"
    # # config_dir="/work/FAC/FGSE/IDYST/gmariet1/gaia/pwiersma/ewatercycle/experiments/config_files"

    # EXP_ID = 'Syn_032b'
    # EXP_ID = 'Syn_162LBa'
    # EXPS = ['Syn_34b','Syn_162LBa']#,'Syn_162LBa']¨
    EXPS = ['Syn_244a','Syn_244b','Syn_162LBa']
    # EXPS = ['Syn_162b','Syn_162LB']
    # EXPS = ['Syn_162a','Syn_162LBa']
    # EXP_ID = 'Syn_162a'
    # EXP_ID = 'Syn_241'
    # EXP_ID = 'Syn_261plus2'
    # EXP_ID = 'Syn_912a'
    plotting = True
    # EXPS = ['Syn_032b','Syn_042']#,'Syn_0425']#,'Syn_2915']
    # EXPS = [EXP_ID]#'Syn_2911',
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

        self.calc_elev_bands(N_bands = 4)
        self.E.bands = list(self.elev_bands)
        self.load_spotpy_SWE()
        self.calc_swe_signatures()
        self.calc_swe_metrics()
        self.load_meteo()
        self.calc_Q_signatures()
        self.calc_Q_metrics()
        self.combine_metrics()
        self.save_metrics()

        
        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.sort_stats('cumtime').print_stats(20)

        # self.load_metrics()
        #plotting
        # self.q_metrics_to_use =  ['KGE_meltseason', 'KGE_meltseason_r','KGE_meltseason_alpha','KGE_meltseason_beta',
        #                  'NSE','Qmean_ME', 'Qmax_ME','t_hfd_ME',
        #                  't_Qinflection_ME','Qmean_meltseason_ME','Qcv_meltseason_ME',
        #                  'BaseFlow_sum_ME',
        #                  'bfi_ME','Qcv_ME','Qamp_ME','t_Qstart_ME','t_Qrise_ME']#'KGE_PeakFilter','KGE_BaseFlow','PeakFilter_sum_ME',
        EXP_ID_translation = {
            # 'Syn_032b':'Full synthetic',
            # 'Syn_2911':'+forcing error',
            # 'Syn_29102':'+soil error (KsatVer = 0.2)',
            # 'Syn_2915':'forcing error +soil error',
            # 'Syn_042':'+forcing error',
            # 'Syn_0425':'+soil error'
        # 'Syn_032b':'Syn',
        #     'Syn_2911':r'Syn + $\epsilon_{forcing}$',
        #     'Syn_29102':'+soil error (KsatVer = 0.2)',
        #     'Syn_2915':r'Syn + $\epsilon_{forcing}$' + r' + $\epsilon_{soil}$',
        #     'Syn_042':r'Syn + $\epsilon_{forcing}$',
        #     'Syn_0425':r'Syn + $\epsilon_{soil}$'
        'Syn_032b':'Fully synthetic',
        'Syn_34b':'Fully Synthetic',
        'Syn_34bb':'Semi-Synthetic',
            'Syn_042':'Semi-Synthetic',
            'Syn_032c':'Quarter synthetic',
            'Syn_912a': 'TEST',
            'Syn_162LB':'Lower Benchmark',
                        'Syn_162LBa':'Lower Benchmark',

            'Syn_162a':'Fully Synthetic',
            'Syn_162b':'Semi-Synthetic',
            'Syn_244a':'Fully Synthetic',
            'Syn_244b':'Semi-Synthetic',
                            # 'Syn_29102':'KsatVer = 0.2',
                                # 'Syn_2915':'KsatVer = 5',
                                }
        self.EXP_ID_translation = EXP_ID_translation

        self.q_metrics_to_use =  ['KGE_meltseason','logKGE_meltseason',# 'KGE_meltseason_r','KGE_meltseason_alpha','KGE_meltseason_beta',
                        'NSE_meltseason','logNSE_meltseason',
                          't_hfd_ME',#'Qmax_ME','Qmean_ME',
                        # 't_Qinflection_ME','Qmean_meltseason_ME','Qcv_meltseason_ME',
                        #  'BaseFlow_sum_ME',#'Qamp_ME',
                        'bfi_ME','t_Qstart_ME',
                        'Qmean_APE', 'Qcv_APE', 'Q95_APE', 'Q5_APE',
                          'Qamp_APE', 'Qmean_meltseason_APE', 'Qcv_meltseason_APE',
                        ]#,'t_Qrise_ME']'NSE','R2','PBIAS','KGE','KGE_PeakFilter','KGE_BaseFlow','Qcv_ME',
        self.swe_metrics_to_use = [
                                    # 'SWE_max_elev_ME',
                                    # 'SWE_max_ME',
                                    # 'melt_sum_ME',
                                    # 'SWE_SWS_ME',
                                    # 't_SWE_max_ME',
                                    # 'SWE_7daymelt_ME',
                                    # 't_SWE_start_ME',
                                    # 't_SWE_end_ME',

                                    # 'melt_sum_elev_ME',
                                    # 'SWE_SWS_elev_ME',
                                    # 't_SWE_max_elev_ME',
                                    # 'SWE_7daymelt_elev_ME',
                                    # 't_SWE_start_elev_ME',
                                    't_SWE_start_ME',
                                    't_SWE_start_grid_ME',

                                    't_SWE_end_ME',
                                    # 't_SWE_end_elev_ME',
                                    't_SWE_end_grid_ME',

                                    # 't_SWE_max_elev_ME',
                                    # 't_SWE_max_ME',
                                    # 't_SWE_max_grid_ME',

                                    'SWE_melt_NSE',
                                    # 'SWE_melt_NSE_elev',
                                    'SWE_melt_NSE_grid',

                                    'SWE_snowfall_NSE',
                                    # 'SWE_snowfall_NSE_elev',
                                    'SWE_snowfall_NSE_grid',
                                    # 'SWE_SPAEF',
                                    # 'SWE_melt_KGE_elev',
                                    # 'SWE_melt_KGE',
                                    # 'SWE_NSE_elev',
                                    # 'SWE_NSE',
                                    # 'SWE_NSE_grid',

                                    'melt_sum_grid_MAPE',
                                    # 'melt_sum_elev_MAPE',
                                    'melt_sum_APE',

                                    # 'SWE_max_APE',
                                    # 'SWE_max_elev_MAPE',
                                    # 'SWE_max_grid_MAPE',

                                    # 'SWE_SWS_APE',
                                    # 'SWE_SWS_elev_MAPE',
                                    # 'SWE_SWS_grid_MAPE',

                                    ]
                                #    'melt_sum_elev_ME',
                                # # 'SWE_SWS_elev_ME',
                                #     # 't_SWE_max_elev_ME',
                                #     'SWE_7daymelt_elev_ME',
                                #     't_SWE_start_elev_ME',
                                #     't_SWE_end_elev_ME',#,#,,
                                #     'SWE_melt_KGE_elev',
                                    # 'SWE_NSE_elev',
                                    # 'SWE_melt_KGE']
                                    # 'SWE_NSE']

        SWE_target_metric = 'melt_sum_elev_MAPE'
        # SWE_target_metric = 'SWE_SWS_

        # compute correlations within Q and SWE metrics and drop the ones we don't need 
        self.compute_QSWE_corr()

        if plotting:
        # for self in LOA_objects.values():
            palette = sns.color_palette('colorblind',n_colors=len(self.swe_metrics_to_use))
            self.swe_palette = {metric: palette[i] for i, metric in enumerate(self.swe_metrics_to_use)}
        
            palette = sns.color_palette('colorblind',n_colors=len(self.q_metrics_to_use))
            self.Q_palette = {metric: palette[i] for i, metric in enumerate(self.q_metrics_to_use)}
            
            # self.compute_metric_corr(self.Qmetrics,self.q_metrics_to_use)
            # self.compute_metric_corr(self.swemetrics,self.swe_metrics_to_use)
            # self.compute_Qsig_corr()
            # self.compute_swesig_elev_corr()

            # self.plot_QSWE_signatures()

            self.plot_QSWE_signatures_stripplot()
            self.plot_best_QSWE_corrs()
    
        # choose a target SWE metric and check which combos of Q metrics work well 
        # q_selection = ['Qamp_ME','Qmean_meltseason_ME',
        #                'Qcv_meltseason_ME','bfi_ME','Qcv_ME',
        #                'Qmean_ME','KGE_meltseason','KGE_meltseason_r']
        q_selection = self.q_metrics_to_use
        swe_selection = self.swe_metrics_to_use
        self.make_Qsig_obs()
        self.calc_corr_combos(Nmax=1)

        # self.dotty_plot('Q95_APE','SWE_SWS_APE')
                 
        LOA_objects[ORIG_ID] = self




# if len(EXPS)>1:
allcorrs_list = []

for EXP_ID in EXPS:
    if 'LB' in EXP_ID: 
        continue
    df = LOA_objects[EXP_ID].QSWE_corr_combos.copy()
    df['EXP_ID'] = EXP_ID_translation[self.ORIG_ID]
    allcorrs_list.append(df)    
allcorrs = pd.concat(allcorrs_list, axis=0)

bestcorrs = allcorrs.groupby(['level_0', 'level_1'])['correlation'].mean().reset_index().sort_values('correlation', ascending=False)

#keep the common string among EXPS list
common = os.path.commonprefix(EXPS)
common_path = join(config_dir,common)
Path.mkdir(Path(common_path),exist_ok = True)

allcorrs.to_csv(join(common_path,'all_correlations.csv'))
bestcorrs.to_csv(join(common_path,'best_correlations.csv'))

# SWE_target_metric
# q_metrics = q_selection + q_combos
# mask = QSWE_corr['level_1'].isin(q_metrics)
# QSWE_corr = QSWE_corr[mask]
# self.SWE_target_metric = 'SWE_SWS_elev_ME'

swe_metrics = [SWE_target_metric]
mask = bestcorrs['level_0'].isin(swe_metrics)
bestcorrs_selection = bestcorrs[mask]

#print the 10 best in a nice way, or make a nice figure showing the top 10
print(bestcorrs_selection.head(10))
# print(bestcorrs_selection.tail(10))
top_q = bestcorrs_selection['level_1'].head(5).values
# top_q = np.append(top_q,['KGE_meltseason'])
selection_corrs = allcorrs[(allcorrs['level_0'].isin(swe_metrics))& (allcorrs['level_1'].isin(top_q))]
#sort selection_corrs based on top_q 
selection_corrs['level_1'] = pd.Categorical(selection_corrs['level_1'],
                                    categories = top_q, ordered = True)
selection_corrs.sort_values('level_1',inplace = True)

f1,ax1 = plt.subplots(figsize = (5,0.5*len(top_q)))
sns.stripplot(selection_corrs, hue = "EXP_ID", y = 'level_1',
            x = 'correlation', orient = 'h',
                dodge = True, ax = ax1,palette = 'colorblind',
                legend = False)
# sns.boxenplot(selection_corrs, hue = "EXP_ID", y = 'level_1', x = 'correlation', orient = 'h',
#                     dodge = True, ax = ax1,palette = 'Blues',
#                     )
ax1.set_xlabel('Spearman rank correlation')
ax1.set_ylabel('Q metric')
ax1.grid()
ax1.axvline(0, color = 'black',  linestyle = '--')
ax1.set_title(f"Top 5 Q metric correlating \n with {SWE_target_metric} ")
# ax1.legend(loc = (1.05,0.5))
# ax1.legend(None)
# plt.suptitle(f"{self.EXP_ID_translation[common]}",y = 1.02, fontsize = 16)
plt.savefig(join(common_path, f"top10_Q_combos_{SWE_target_metric}.png"),
            dpi = 300, bbox_inches = 'tight')
    

#%%
# Q_selection = pd.Categorical(selection_corrs['level_1'], categories=selection_corrs['level_1'].unique(), ordered=True)
# Q_selection = Q_selection.categories[:5].tolist()
# Q_selection = ['KGE_meltseason', 't_hfd_ME+Qmean_meltseason_ME']
# Q_selection = ['KGE_meltseason']
Q_selection = ['NSE_meltseason']
Q_chosen = 'NSE_meltseason'
# Q_selection = ['t_hfd_ME+Qmean_meltseason_ME']
# SWE_chosen = SWE_target_metric
# SWE_chosen = 'SWE_NSE_elev'
# SWE_chosen = 'melt_sum_elev_MAPE'
# SWE_targets = ['melt_sum_elev_MAPE','SWE_NSE_elev','SWE_melt_KGE_elev','SWE_melt_NSE',
#                't_SWE_start_ME','t_SWE_start_elev_ME']
SWE_targets = self.swe_metrics_to_use
# SWE_chosen = 'SWE_SWS_elev_MAPE'
# SWE_chosen = 'SWE_melt_KGE'
# SWE_chosen = 'SWE_melt_NSE'

for self in LOA_objects.values():
    # self = LOA_objects[EXPS[0]]
    # Lower_Benchmark = None
    Lower_Benchmark = LOA_objects[EXPS[2]] #None

    if 'LB' in self.EXP_ID:
        print(f"Skipping {self.EXP_ID} as it is a lower benchmark")
        continue

    #Determine Q posterior, LB and UB IDs and the corresponding SWE metrics 
    SWE_PRI_dic = {}
    for SWE_chosen in SWE_targets:
        SWE_PRI_dic[SWE_chosen] = {}    
        SWE_target_results_list = []

        for i, Q_chosen in enumerate(Q_selection):
            if '+' in Q_chosen:
                Q_chosen_ = Q_chosen.split('+')
            else:
                Q_chosen_ = [Q_chosen]

            LOA_metrics = self.calculate_LOA_metrics( Q_chosen_)
            LOA_checks = self.perform_LOA_checks(Q_chosen_, LOA_metrics,flat_percentage = 1)
            SWE_results = self.gather_SWE_results(LOA_checks, Q_chosen,SWE_chosen,
                                                    Lower_Benchmark=Lower_Benchmark)
            SWE_target_results_list.append(SWE_results)

        SWE_target_results_all = pd.concat(SWE_target_results_list, axis=0).reset_index()
        SWE_PRI_dic[SWE_chosen]['SWE_results'] = SWE_target_results_all

    # Rerun UB with perfect rfcf, or load from file, or simply use Q with UB ids (not perfect rfcf)
    rerun_UB = False #True for perfect rfcf 
    for SWE_chosen in SWE_targets:
        if rerun_UB:
            #Run Upper benchmark again for Q         
            UB_dir = join(self.LOA_DIR, f"UB_{SWE_chosen}")
            Path.mkdir(Path(UB_dir),exist_ok = True)
            SWE_results = SWE_PRI_dic[SWE_chosen]['SWE_results']
            for year in range(self.START_YEAR, self.END_YEAR+1):
                UB_file = join(UB_dir,f'UB_pars_{year}.csv')
                if not os.path.isfile(UB_file):
                    print(f"Running UB with perfect sfcf for {year}")
                    UB_ids = SWE_results[(SWE_results['category'] == 'UB') & (SWE_results['year'] == year)]['index']
                    #load pars related to these IDs
                    UB_pars = self.pars[year].loc[UB_ids]
                    #Replace rfcf with the true rfcf 
                    UB_pars['rfcf'] = self.trueparams['rfcf']
                    #save these ids to a text file 
                    UB_pars.to_csv(UB_file)
                    #run the model
                    Postruns(self.config, calibration_purpose='UB', single_year=year, UB_dir = UB_dir)
            else:
                #Just load Q from the folder
                pass 
            Q_UB = {}
            print(f"Loading Q_UB for {SWE_chosen} from files")
            for year in range(self.START_YEAR, self.END_YEAR + 1):
                UB_yearly_dir = join(UB_dir, f"{year}")
                Q_file = join(UB_yearly_dir, f"{self.config['BASIN']}_Q.csv")
                Q_UBy = pd.read_csv(Q_file, index_col=0, parse_dates=True)
                #drop the obs column
                Q_UBy.drop(columns = 'obs',inplace = True)
                #rename all columns to just their ID (which is the last ms0_* part)
                Q_UBy.columns = ["ms0_"+col.split('_')[-1] for col in Q_UBy.columns]
                Q_UB[year] = Q_UBy

            #     #plot test 
                # idx = 0
                # sslice = slice(f"{year}-03-01",f"{year}-09-30")
                # qobs = self.Qobs.loc[sslice]
                # qub = Q_UB[year][sslice]
                # qsim = self.Q.loc[:,qub.columns].loc[sslice]
                # f1,ax1 = plt.subplots()
                # qobs.plot(color = 'black', label = 'obs', ax = ax1,zorder = 100)
                # qub[qub.columns].plot(color = 'red', label = 'UB', ax = ax1, alpha = 0.2,zorder = 50)
                # qsim[qub.columns].plot(color = 'blue', label = 'sim', ax = ax1,alpha = 0.5)
                # ax1.legend([])

        else:
            print(f"Loading Q_UB for {SWE_chosen} from files with incorrect rfcf")
            Q_UB = {}
            SWE_results = SWE_PRI_dic[SWE_chosen]['SWE_results']
            for year in range(self.START_YEAR, self.END_YEAR + 1):
                UB_ids = SWE_results[(SWE_results['category'] == 'UB') & (SWE_results['year'] == year)]['index']
                #from self.Q take these IDs and crop the years
                timeslice = slice(f"{year-1}-10-01",f"{year}-09-30")
                Q_UBy = self.Q.loc[timeslice,UB_ids]
                Q_UB[year] = Q_UBy
        SWE_PRI_dic[SWE_chosen]['Q_UB'] = Q_UB

    def calc_PRI(rank_stack,year,cat):
        # 
        rs = rank_stack[rank_stack['Year'] == year]
        selection_rank = rs[rs['Category'] == cat]['Rank']
        best_median = len(selection_rank.unique())/2
        N = SEL[SEL['year'] == year][SEL['category']=='Prior'].shape[0]
        worst_median = N / 2
        # PRI = worst_median - (selection_rank.median() - best_median)
        PRI = 1- selection_rank.median() /(worst_median - best_median)
        return PRI

    def calc_median_normalized_rank(rank_stack, year, cat):
        #just number betwee 0 and 1, where 1 is the best and 0 is the worst 
        rs = rank_stack[rank_stack['Year'] == year]
        selection_rank = rs[rs['Category'] == cat]['Rank']
        N = SEL[SEL['year'] == year][SEL['category']=='Prior'].shape[0]
        median_normalized_rank = 1- (selection_rank.median() / N)
        return median_normalized_rank

    def calc_median_rank(rank_stack, year, cat):
        #just number betwee 0 and 1, where 1 is the best and 0 is the worst 
        rs = rank_stack[rank_stack['Year'] == year]
        selection_rank = rs[rs['Category'] == cat]['Rank']
        median_rank = selection_rank.median() 
        return median_rank

    ## %%
    #for each SWE metric, calculate rank_stacks and PRI's
    for SWE_chosen in SWE_targets:
        #frst for SWE 
        print(f"Calculating PRI for {SWE_chosen}")
        SEL = SWE_PRI_dic[SWE_chosen]['SWE_results'][SWE_PRI_dic[SWE_chosen]['SWE_results']['Q_metric'] == Q_chosen]

        rank_stack_swe = pd.DataFrame(columns = ['Rank','Year','Category'])
        for year in range(self.START_YEAR, self.END_YEAR+1):
            all_metrics = self.metrics[year][SWE_chosen]
            ascending = False if (('KGE' in SWE_chosen) or ('NSE' in SWE_chosen) or ('SPAEF' in SWE_chosen)) else True
            all_ranks = all_metrics.rank(ascending = ascending)#, na_option='bottom').astype(int)
            post_ranks = all_ranks.loc[SEL[(SEL['category'] == 'Posterior') & (SEL['year'] == year)]['index']]
            if not Lower_Benchmark == None:
                LB_metrics= Lower_Benchmark.metrics[year][SWE_chosen]
                interp_indices = pd.DataFrame(columns = ['All_index'])
                for met in LB_metrics.index:
                    if not np.isnan(LB_metrics.loc[met]):
                        closest_index = (all_metrics - LB_metrics.loc[met]).abs().idxmin()
                        interp_indices.loc[met, 'All_index'] = closest_index
                LB_ranks = all_ranks.loc[interp_indices['All_index']]
                rank_stack_swe = pd.concat([rank_stack_swe, pd.DataFrame({'Rank':LB_ranks,'Year':year,'Category':'LB'})],axis = 0)
    # rank_stack = pd.concat([rank_stack, pd.DataFrame({'Rank':all_ranks,'Year':year,'Category':'Prior'})],axis = 0)
            rank_stack_swe = pd.concat([rank_stack_swe, pd.DataFrame({'Rank':post_ranks,'Year':year,'Category':'Posterior'})],axis = 0)
        SWE_PRI_dic[SWE_chosen]['rank_stack_swe'] = rank_stack_swe

        LB_PRI_values = {}
        Posterior_PRI_values = {}
        for year in range(self.START_YEAR, self.END_YEAR+1):
            # LB_PRI_value = calc_PRI(rank_stack_swe, year, 'LB')
            # LB_PRI_value = calc_median_normalized_rank(rank_stack_swe, year, 'LB')
            LB_PRI_value = calc_median_rank(rank_stack_swe, year, 'LB')
            LB_PRI_values[year] = LB_PRI_value
            print(f"PRI for year {year}: {LB_PRI_value}")
            # Posterior_PRI_value = calc_PRI(rank_stack_swe, year, 'Posterior')
            # Posterior_PRI_value = calc_median_normalized_rank(rank_stack_swe, year, 'Posterior')
            Posterior_PRI_value = calc_median_rank(rank_stack_swe, year, 'Posterior')
            Posterior_PRI_values[year] = Posterior_PRI_value
            print(f"PRI for year {year}: {Posterior_PRI_value}")

        #put LB_PRI_values and Posterior_PRI_values in a melt df and plot them
        PRI_values = pd.DataFrame({'year':list(LB_PRI_values.keys())*2,
                    'category':['Posterior']*len(LB_PRI_values)+['LB']*len(LB_PRI_values),
                    'PRI':list(Posterior_PRI_values.values())+list(LB_PRI_values.values())})
        SWE_PRI_dic[SWE_chosen]['PRI_SWE'] = PRI_values

        #then for Q
        Q_target_results_list = []
        for year in range(self.START_YEAR, self.END_YEAR+1):

            all_ids = SEL[SEL['year'] == year]['index'][SEL['category'] == 'Prior']
            post_ids = SEL[(SEL['category'] == 'Posterior') & (SEL['year'] == year)]['index']   
            UB_ids = SEL[(SEL['category'] == 'UB') & (SEL['year'] == year)]['index']
            
            # Create separate DataFrames for each category
            all_df = self.metrics[year][Q_chosen].loc[all_ids].to_frame().reset_index()
            all_df['year'] = year
            all_df['category'] = 'Prior'
            
            # Create DataFrame for Posterior category (if any IDs exist)
            if len(post_ids) > 0:
                post_df = self.metrics[year][Q_chosen].loc[post_ids].to_frame().reset_index()
                post_df['year'] = year
                post_df['category'] = 'Posterior'
                all_df = pd.concat([all_df, post_df], axis=0)
            
            # Create DataFrame for UB category 
            ub_metric = pd.DataFrame(index = UB_ids,columns = [Q_chosen])
            ub_metric = ub_metric.astype(float)
            for i, ID in enumerate(UB_ids):
                #NSE_meltseason 
                if Q_chosen == 'NSE_meltseason':
                    daterange = pd.date_range(start = f"{year-1}-10-01", end = f"{year}-09-30")
                    q_obs = self.Qobs.loc[daterange].squeeze()
                    q_sim = SWE_PRI_dic[SWE_chosen]['Q_UB'][year][ID]
                    ub_metric.loc[ID,Q_chosen] = he.nse(q_sim.loc[q_sim.index.month.isin([4, 5, 6, 7])],q_obs.loc[q_obs.index.month.isin([4, 5, 6, 7])])
                else:
                    print("WE HAVE A PROBLEM")
                    sys.exit()
                    # ub_metric = self.calc_NSE_meltseason(Qobs,Qsim)
                # ub_df[Q_chosen] = ub_metric[Q_chosen]
                ub_metric['year'] = year
                ub_metric['category'] = 'UB'
                ub_metric[Q_chosen] = ub_metric[Q_chosen].values
            all_df = pd.concat([all_df, ub_metric.reset_index()], axis=0)

            # Add the All runs data
            Q_target_results_list.append(all_df)

        # Concatenate all DataFrames
        Q_target_results_all = pd.concat(Q_target_results_list, axis=0).set_index('index').reset_index()
        SWE_PRI_dic[SWE_chosen]['Q_results'] = Q_target_results_all#[Q_target_results_all['category'] == 'UB']#.drop(columns=['category'])

        #calc ranks 
        rank_stack = pd.DataFrame(columns = ['Rank','Year','Category'])
        for year in range(self.START_YEAR, self.END_YEAR+1):
            all_metrics = self.metrics[year][Q_chosen]
            ascending = False if (('KGE' in Q_chosen) or ('NSE' in Q_chosen)) else True
            all_ranks = all_metrics.rank(ascending = ascending,na_option = 'keep').astype('Int64')
            # UB_ranks = all_ranks.loc[SEL[(SEL['category'] == 'UB') & (SEL['year'] == year)]['index']]
            # LB_metrics= Lower_Benchmark.metrics[year][SWE_chosen]
            UB_metrics = Q_target_results_all[(Q_target_results_all['category'] == 'UB')& (Q_target_results_all['year'] == year)][Q_chosen]
            interp_indices = pd.DataFrame(index = UB_metrics.index,columns = ['All_index'])
            for met in UB_metrics.index:
                if not np.isnan(UB_metrics.loc[met]):
                    closest_index = (all_metrics - UB_metrics.loc[met]).abs().idxmin()
                    interp_indices.loc[met, 'All_index'] = closest_index
            UB_ranks = all_ranks.loc[interp_indices['All_index']]
            UB_ranks.index = UB_metrics.index
            
            # UB_ranks =pd.Series(index = interp_indices.index,
            #             data = all_ranks.loc[interp_indices['All_index']])
                
            rank_stack = pd.concat([rank_stack, pd.DataFrame({'Rank':UB_ranks,'Year':year,'Category':'UB'})],axis = 0)

        UB_PRI_values = {}
        for year in range(self.START_YEAR, self.END_YEAR+1):
            # UB_PRI_value = calc_PRI(rank_stack, year, 'UB')
            # UB_PRI_value = calc_median_normalized_rank(rank_stack, year, 'UB')
            UB_PRI_value = calc_median_rank(rank_stack, year, 'UB')
            UB_PRI_values[year] = UB_PRI_value
            print(f"PRI for year {year}: {UB_PRI_value}")
        
        #put LB_PRI_values and Posterior_PRI_values in a melt df and plot them
        PRI_values = pd.DataFrame({'year':list(UB_PRI_values.keys()),
                            'category':['UB']*len(UB_PRI_values),
                            'PRI':list(UB_PRI_values.values())})
        SWE_PRI_dic[SWE_chosen]['PRI_Q'] = PRI_values
    self.SWE_PRI_dic = SWE_PRI_dic

#plot settings 
categories = ['Prior','Posterior','UB','LB']
alphas = dict(zip(categories, [0.2, 1, 0.3, 0.8]))
markersize = dict(zip(categories, [2, 4, 4, 2]))
from pypalettes import load_cmap
colormap = load_cmap("Egypt")       
colors_list = [colormap(i) for i in np.linspace(0, 1, 4)]

# Alternatively, you can pick specific positions from the colormap:
color1 = colormap(0.1)  # Color at 10% of the colormap
color2 = colormap(0.3)  # Color at 30% of the colormap
color3 = colormap(0.7)  # Color at 70% of the colormap
color4 = colormap(0.9)  # Color at 90% of the colormap        
# colormap = sns.palettes.color_palette('colorblind',4)
# Create a discrete colormap with specific number of levels
discrete_cmap = load_cmap("Egypt")  # For example, 10 distinct colors
colormap = [discrete_cmap(3/4), discrete_cmap(2/4), discrete_cmap(1/4), discrete_cmap(0/4)]

colors = dict(zip(categories,colormap))    
# colors['Posterior'] = colormap[2]
colors['Prior'] = 'grey'
# colors = dict(zip(categories,colors_list))
zorder= dict(zip(categories,[1,20,3,4]))

from matplotlib.lines import Line2D
dothandles =dict(zip(categories,
    [Line2D([0],[0],color = colors[cat], marker = 'o', markersize = 10, linestyle = '') for cat in categories]))
linehandles = dict(zip(categories,
    [Line2D([0],[0],color = colors[cat], linestyle = '-') for cat in categories]))                  
patch_handles = dict(zip(categories,
    [Line2D([0],[0],color = colors[cat], linestyle = '-',linewidth = 5, alpha = 0.2) for cat in categories]))
dothandle_obs = Line2D([0],[0],color = 'black', marker = 'o', 
                       markersize = 10, linestyle = '')
linehandle_obs = Line2D([0],[0],color = 'black', linestyle = '--')
patchhandle_obs = Line2D([0],[0],color = 'black', linestyle = '-',linewidth = 5, alpha = 0.2)
# def translate_SWE_metric_name(metric_name,keep_suffix = True):
#     new_name = metric_name
#     elev = False
#     grid = False
#     if 'elev' in metric_name:
#         new_name = metric_name.replace('_elev', '')
#         elev = True
#     elif 'grid' in metric_name:
#         new_name = metric_name.replace('_grid', '')
#         grid = True
        
#     readable_dic = {
#         'melt_sum': 'Accumulation',
#         'SWE_melt_KGE': 'Melt KGE',
#         'SWE_melt_NSE': 'Melt NSE',
#         'SWE_NSE': 'SWE NSE',
#         'SWE_SWS': 'SWS',
#         't_SWE_start': 'Start of SWE timing',
#         't_SWE_end': 'End of SWE timing',
#         't_SWE_max': 'Maximum SWE timing',
#         'SWE_snowfall_NSE': 'Snowfall NSE',
#         'SWE_SPAEF': 'Spatial Efficiency'  # Fixed this line
#     }
#     for key in readable_dic.keys():
#         if key in new_name:
#             # print(key)
#             new_name = new_name.replace(key, readable_dic[key])
#     if '_' in new_name:
#         new_name = new_name.replace('_'+new_name.split('_')[-1], ' error')
#     if keep_suffix:
#         if elev:
#             new_name = 'ELEV-' + new_name
#         elif grid:
#             new_name = 'GRID-' + new_name
#         else:
#             new_name = 'AGG-' + new_name
#     # if '_' in new_name:
#     #     # new_name = new_name.replace('_', ' ')
#     #     new_name = new_name.split('_')[0]
#     return new_name
#%% numbers
for self in LOA_objects.values():
    if 'LB' in self.EXP_ID:
        continue
    SWE_PRI_dic = self.SWE_PRI_dic
    print(f"Results for {self.EXP_ID_translation[self.ORIG_ID]}")
    posterior_Q = SWE_PRI_dic[SWE_chosen]['Q_results'][SWE_PRI_dic[SWE_chosen]['Q_results']['category'] == 'Posterior']
    mean_posterior_Q = posterior_Q[Q_chosen].mean()
    print("Posterior mean Q metric: ", mean_posterior_Q)
    std_posterior_Q = posterior_Q[Q_chosen].std()
    print("Posterior std Q metric: ", std_posterior_Q)

    annual_min = posterior_Q.groupby('year').min()[Q_chosen]
    annual_max = posterior_Q.groupby('year').max()[Q_chosen]
    annual_range = annual_max - annual_min
    print("Posterior annual range Q metric: ", annual_range.mean())
    print("Posterior annual range Q metric std: ", annual_range.std())

    #same for prior Q metric ranges 
    prior_Q = SWE_PRI_dic[SWE_chosen]['Q_results'][SWE_PRI_dic[SWE_chosen]['Q_results']['category'] == 'Prior']
    mean_prior_Q = prior_Q.groupby('year')[Q_chosen].mean()
    print("Prior mean Q metric: ", mean_prior_Q.mean())
    print("Prior std Q metric: ", mean_prior_Q.std())

    annual_min = prior_Q.groupby('year').min()[Q_chosen]
    annual_max = prior_Q.groupby('year').max()[Q_chosen]
    annual_range = annual_max - annual_min
    print("Prior minimum Q metric: ", annual_min.mean())
    print("Prior maximum Q metric: ", annual_max.mean())
    print("Prior annual range Q metric: ", annual_range.mean())
    print("Prior annual range Q metric std: ", annual_range.std())

    SWE_chosen = 'SWE_melt_NSE'
    rstack = SWE_PRI_dic[SWE_chosen]['rank_stack_swe']
    posterior_SWE = rstack[rstack['Category'] == 'Posterior']
    median_ranks = posterior_SWE.groupby('Year')['Rank'].median()
    print(f"Posterior mean median rank for {SWE_chosen}: ", median_ranks.mean())

    SWE_chosen = 'melt_sum_APE'
    metrics = SWE_PRI_dic[SWE_chosen]['SWE_results']
    # self.swe_signatures[year]['sim']['melt_sum'][id]
    obs_accumulation= [self.swe_signatures[year]['obs']['melt_sum'] for year in metrics['year'].unique()]
    obs_accumulation = pd.DataFrame(obs_accumulation, columns=['melt_sum'], index=metrics['year'].unique())
    posterior_SWE = metrics[metrics['category'] == 'Posterior']
    posterior_SWE['melt_sum'] = np.nan
    posterior_SWE['PE'] = np.nan
    for idx in posterior_SWE.index:
        year = posterior_SWE.loc[idx, 'year']
        id = posterior_SWE.loc[idx, 'index']
        sim_melt_sum= self.swe_signatures[year]['sim']['melt_sum'][id].item()
        posterior_SWE.loc[idx, 'melt_sum'] = sim_melt_sum
        posterior_SWE.loc[idx, 'PE'] = (obs_accumulation.loc[year, 'melt_sum'] - sim_melt_sum) / obs_accumulation.loc[year, 'melt_sum']
    mean_PE = posterior_SWE.groupby('year')['PE'].mean()
    print(f"Posterior mean PE for {SWE_chosen}: ", mean_PE.mean())
    print(f"Posterior std PE for {SWE_chosen}: ", mean_PE.std())

##%%

    # Plot toggles
    plot_SWE_metrics = False
    plot_Q_metrics = False
    plot_SWE_Q_combo = True
    plot_EGU = False

    # def add_suffix(metric):
    #     if ('SWE_melt' in metric) or ('SWE_snowfall' in metric):
    #         return 'NSE [-]'
    #     else:
    #         if 't_SWE' in metric:
    #             if ('elev' in metric) or ('grid' in metric):
    #                 return 'MAE [days]'
    #             else:
    #                 return 'ME [days]'
    #         else:
    #             if ('elev' in metric) or ('grid' in metric):
    #                 return 'MAPE [%]'
    #             else:   
    #                 return 'APE [%]'
    def add_suffix(metric):
        if ('SWE_melt' in metric) or ('SWE_snowfall' in metric):
            return '[-]'
        else:
            if 't_SWE' in metric:
                if ('elev' in metric) or ('grid' in metric):
                    return '[days]'
                else:
                    return '[days]'
            else:
                if ('elev' in metric) or ('grid' in metric):
                    return '[%]'
                else:   
                    return '[%]'


    from matplotlib.lines import Line2D

    def plot_swe_metrics(SWE_chosen, SWE_cats, PRI_SWE, alphas, colors, zorder, exp_title):
        f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        plt.subplots_adjust(hspace=0.3)

        for category, alpha in alphas.items():
            subset = SWE_cats[category]
            if category == 'LB':
                sns.boxenplot(data=subset, x='year', y=SWE_chosen, hue='category', ax=ax1,
                            alpha=alpha, orient='v', palette=colors, zorder=zorder[category])
            else:
                sns.stripplot(data=subset, x='year', y=SWE_chosen, hue='category', ax=ax1,
                            dodge=False, alpha=alpha, orient='v', s=markersize[category],
                            palette=colors, zorder=zorder[category])

        custom_lines = [Line2D([0], [0], color=colors[cat], marker='o', markersize=10, linestyle='') for cat in SWE_cats]
        ax1.legend(custom_lines, SWE_cats.keys(), loc=[1.05, 0.5])
        ax1.set_ylim((0.00, 1) if 'KGE' in SWE_chosen or 'NSE' in SWE_chosen else (ax1.get_ylim()[1]/2, 0))
        ax1.grid()
        ax1.set_ylabel(SWE_chosen)
        ax1.set_title('SWE metric results')

        sns.pointplot(data=PRI_SWE, x='year', y='PRI', hue='category', ax=ax2,
                    linestyles='--', palette=colors)
        posterior_avg = PRI_SWE[PRI_SWE['category'] == 'Posterior']['PRI'].mean()
        LB_avg = PRI_SWE[PRI_SWE['category'] == 'LB']['PRI'].mean()
        ax2.set_title(f"Posterior Avg = {posterior_avg:.2f} \n LB Avg = {LB_avg:.2f}")
        ax2.set_ylabel('Posterior Median Rank [-]')
        ax2.axhline(1500, color='black', linestyle='--')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_ylim(ax2.get_ylim()[1], 1)
        ax2.grid()
        ax2.legend([])
        f1.suptitle(exp_title, y=1, fontsize=16)


    def plot_q_metrics(Q_chosen, SWE_chosen, Q_cats, PRI_Q, alphas, colors, dothandles, exp_title):
        f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        for good, alpha in alphas.items():
            if good == 'LB':
                continue
            subset = Q_cats[good]
            sns.stripplot(data=subset, x='year', y=Q_chosen, ax=ax1, orient='v', hue='category',
                        s=markersize[good], palette=colors, alpha=alpha)
            if good == 'UB':
                median_UB = subset.groupby('year')[Q_chosen].median()
                sns.pointplot(median_UB, ax=ax1, color=colors['UB'], linestyles='--')

        ax1.set_ylim(0.0, 1)
        ax1.grid()
        ax1.set_title(f"Streamflow metric results for {SWE_chosen}")
        handles = [dothandles['Prior'], dothandles['Posterior'], dothandles['UB']]
        labels = ['Prior', 'Posterior', 'Upper Benchmark']
        ax1.legend(handles, labels, loc=[0.8, 1.1])

        sns.pointplot(data=PRI_Q, x='year', y='PRI', hue='category', ax=ax2,
                    linestyles='--', palette=colors)
        UB_avg = PRI_Q[PRI_Q['category'] == 'UB']['PRI'].mean()
        ax2.set_title(f"UB Average = {UB_avg:.2f}")
        ax2.set_ylabel('Posterior Rank Improvement [-]')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.grid()
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.legend([])
        f1.suptitle(exp_title, y=1, fontsize=16)


    def plot_swe_q_combo(SWE_chosen, Q_chosen, SWE_cats, Q_cats, alphas, colors, zorder, dothandles, exp_title):
        f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5), sharex=True)

        max1swemetric = ('KGE' in SWE_chosen or 'NSE' in SWE_chosen or 'SPAEF' in SWE_chosen)

        # Plot Streamflow Metrics (Q_chosen)
        lowest_Q = 1000
        for good, alpha in alphas.items():
            if good in ['LB', 'UB']:
                continue
            subset = Q_cats[good]
            sns.stripplot(data=subset, x='year', y=Q_chosen, ax=ax1, orient='v', hue='category',
                        s=markersize[good], palette=colors, alpha=alpha,
                        jitter=0.3, edgecolor='black', linewidth=0.5 if good == 'Posterior' else 0.1)
            if good == 'Posterior' and np.min(subset[Q_chosen]) < lowest_Q:
                lowest_Q = np.min(subset[Q_chosen])
            # if good =='Posterior':
            #     best = subset.groupby('year').max()
            #     sns.stripplot(best,x='year', y=Q_chosen, ax=ax1, orient='v', hue='category',
            #             s=markersize[good]*1.2, palette = 'gist_ncar',
            #             jitter=0.3, edgecolor='black', linewidth=0.5,zorder = 1e7)

        ax1.grid(alpha=0.5)
        # ax1.set_title("Streamflow results & Posterior ensemble selection")
        if exp_title == 'Fully Synthetic':
            ax1.legend([dothandles['Prior'], dothandles['Posterior']],
                    ['5000 Prior runs', '50 Posterior runs'], loc=[-0.1, 1.1])
        else:
            ax1.legend([],edgecolor ='white')
        # ax1.legend([dothandles['Prior'], dothandles['Posterior']], 
        #            ['5000 Prior runs', '50 Posterior runs'], loc=[0.81, 1.15])
        ax1.set_ylim(lowest_Q, 1)
        # ax1.set_ylabel("NSE [-]")
        ax1.set_ylabel(r"$E_{Q-NSE}$ [-]")

        # Plot SWE Metrics (SWE_chosen)
        lowest_SWE = 1000 if max1swemetric else 0
        for category, alpha in alphas.items():
            if category in ['LB', 'UB']:
                continue
            subset = SWE_cats[category]
            sns.stripplot(data=subset, x='year', y=SWE_chosen, hue='category', ax=ax2,
                        dodge=True, jitter=0.3,  # Add jitter to the points
                        alpha=alpha, orient='v', s=markersize[category],
                        palette=colors, zorder=zorder[category],
                        edgecolor='black', linewidth=0.5 if category == 'Posterior' else 0.1)
            if category == 'Posterior':
                val = np.min(subset[SWE_chosen]) if max1swemetric else np.max(subset[SWE_chosen])
                if (val < lowest_SWE and max1swemetric) or (val > lowest_SWE and not max1swemetric):
                    lowest_SWE = val
            #     # Get the best SWE values corresponding to the best Q indices
            #     best_swe = subset.loc[subset.apply(lambda row: (row['year'], row['index']) in zip(best.index, best['index']), axis=1)]
            #     sns.stripplot(data=best_swe, x='year', y=SWE_chosen, ax=ax2, orient='v', hue='category',
            #                 s=markersize[category]*1.2, palette='gist_ncar',
            #                 jitter=0.3, edgecolor='black', linewidth=0.5,zorder = 1e7)

        ax2.legend([],edgecolor ='white')
        ax2.set_ylim((lowest_SWE, 1) if max1swemetric else (lowest_SWE, 0))
        ax2.grid(alpha=0.5)
        if 'grid' in SWE_chosen:
            ax2.set_ylabel(f"GRID \n {translate_SWE_metric_name(SWE_chosen)} {add_suffix(SWE_chosen)}")
            ax2.set_ylabel(fr"$E_{{GRID}}^{{{translate_SWE_metric_name(SWE_chosen)}}}$ {add_suffix(SWE_chosen)}")
        elif 'elev' in SWE_chosen:
            ax2.set_ylabel(f"ELEV \n {translate_SWE_metric_name(SWE_chosen)} {add_suffix(SWE_chosen)}")
        else:
            ax2.set_ylabel(f"AGG \n {translate_SWE_metric_name(SWE_chosen)} {add_suffix(SWE_chosen)}")
            ax2.set_ylabel(fr"$E_{{AGG}}^{{{translate_SWE_metric_name(SWE_chosen)}}}$ {add_suffix(SWE_chosen)}")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_xlabel(None)
        # ax2.set_title('Evaluation of posterior SWE scenarios')
        for ax in [ax1,ax2]:
                for spine in ax.spines.values():
                    spine.set_color('tab:blue' if exp_title =='Fully Synthetic' else 'tab:orange')
                    spine.set_linewidth(1.5)

        fontcol = 'tab:blue' if exp_title =='Fully Synthetic' else 'tab:orange'
        text = f"<{exp_title}>"
        fig_text(
            x=0.53, # position on x-axis
            y=0.96, # position on y-axis
            ha='center',
            s=text,
            fontsize=18,
            ax=ax1,
            highlight_textprops=[{'color':fontcol,'fontweight':'bold'}]
        )
        # Set the overall title
        # f1.suptitle(exp_title, y=1, fontsize=16)

        # Adjust layout
        plt.tight_layout()
        f1.savefig(join(self.FIGDIR,f"priorposteriorplot_{SWE_chosen}_{Q_chosen}_Q_SWE_.png"),
                bbox_inches='tight', dpi=400)
        # f1.savefig(join("/home/pwiersma/scratch/Figures/Paper1",f"Fig3_{self.ORIG_ID}.png"), dpi=500, bbox_inches='tight')
        plt.show()

    def plot_swe_q_combo_EGU(SWE_chosen, Q_chosen, SWE_cats, Q_cats, alphas, colors, zorder, dothandles, exp_title):
        f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,7.6), sharex=True)
        max1swemetric = ('KGE' in SWE_chosen or 'NSE' in SWE_chosen or 'SPAEF' in SWE_chosen)

        # Plot Streamflow Metrics (Q_chosen)
        lowest_Q = 1000
        for good, alpha in alphas.items():
            if good in ['LB', 'UB']:
                continue
            subset = Q_cats[good]
            sns.stripplot(data=subset, x='year', y=Q_chosen, ax=ax1, orient='v', hue='category',
                        s=markersize[good], palette=colors, alpha=alpha,
                        jitter=0.3, edgecolor='black', linewidth=0.5 if good == 'Posterior' else 0.1)
            if good == 'Posterior' and np.min(subset[Q_chosen]) < lowest_Q:
                lowest_Q = np.min(subset[Q_chosen])

        ax1.grid(alpha=0.5)
        ax1.set_title("Streamflow results \n Posterior = 50 best runs")
        if exp_title == 'Fully Synthetic':
            ax1.legend([dothandles['Prior'], dothandles['Posterior']],
                    ['5000 Prior runs', '50 Posterior runs'], 
                    # loc=[0.8, 1.1])
                    loc = 'lower right',
                    fontsize = 13)
        else:
            ax1.legend([],edgecolor ='white')
        # ax1.set_ylim(lowest_Q, 1)
        ax1.set_ylim(0.8,1)
        ax1.set_ylabel("NSE [-]")

        # Plot SWE Metrics (SWE_chosen)
        lowest_SWE = 1000 if max1swemetric else 0
        for category, alpha in alphas.items():
            if category in ['LB', 'UB']:
                continue
            subset = SWE_cats[category]
            sns.stripplot(data=subset, x='year', y=SWE_chosen, hue='category', ax=ax2,
                        dodge=True, jitter=0.3,  # Add jitter to the points
                        alpha=alpha, orient='v', s=markersize[category],
                        palette=colors, zorder=zorder[category],
                        edgecolor='black', linewidth=0.5 if category == 'Posterior' else 0.1)
            if category == 'Posterior':
                val = np.min(subset[SWE_chosen]) if max1swemetric else np.max(subset[SWE_chosen])
                if (val < lowest_SWE and max1swemetric) or (val > lowest_SWE and not max1swemetric):
                    lowest_SWE = val

        ax2.legend([],edgecolor ='white')
        ax2.set_xlabel(None)
        # ax2.set_ylim((lowest_SWE, 1) if max1swemetric else (lowest_SWE, 0))
        ax2.set_ylim(0.5,1)
        ax2.grid(alpha=0.5)
        if 'grid' in SWE_chosen:
            ax2.set_ylabel(f"Distributed \n {translate_SWE_metric_name(SWE_chosen)} [mm]")
        else:
            ax2.set_ylabel(f"Catchment-wide \n {translate_SWE_metric_name(SWE_chosen)} NSE [-]")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_title('Evaluation of posterior SWE scenarios')

        for ax in [ax1,ax2]:
            for spine in ax.spines.values():
                spine.set_color('tab:blue' if exp_title =='Fully Synthetic' else 'tab:orange')
                spine.set_linewidth(1.5)

        fontcol = 'tab:blue' if exp_title =='Fully Synthetic' else 'tab:orange'
        text = f"<{exp_title}>"
        fig_text(
            x=0.53, # position on x-axis
            y=0.98, # position on y-axis
            ha='center',
            s=text,
            fontsize=20,
            ax=ax1,
            highlight_textprops=[{'color':fontcol,'fontweight':'bold'}]
        )

        # Set the overall title
        # f1.suptitle(exp_title, y=0.94, fontsize=20)
        # f1.savefig(join(self.FIGDIR,f"priorposteriorplot_{SWE_chosen}_{Q_chosen}_Q_SWE_.png"))
        f1.savefig(join(self.FIGDIR,f"priorposteriorplot_POSTERnarrow_{SWE_chosen}_{Q_chosen}_Q_SWE_.png"), dpi=300, bbox_inches='tight')
        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_swe_comparison(SWE_chosen, LOA_obj_a, LOA_obj_b, alphas, colors, zorder, exp_title_a, exp_title_b):
        """
        Plot SWE results for two LOA objects in separate subplots for comparison.
        
        Parameters:
        -----------
        SWE_chosen : str
            The SWE metric to plot
        LOA_obj_a : LOA object
            First LOA object (e.g., Syn_244a)
        LOA_obj_b : LOA object  
            Second LOA object (e.g., Syn_244b)
        alphas : dict
            Dictionary of alpha values for different categories
        colors : dict
            Dictionary of colors for different categories
        zorder : dict
            Dictionary of zorder values for different categories
        exp_title_a : str
            Title for first experiment (e.g., 'Fully Synthetic')
        exp_title_b : str
            Title for second experiment (e.g., 'Semi-Synthetic')
        """
        f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        max1swemetric = ('KGE' in SWE_chosen or 'NSE' in SWE_chosen or 'SPAEF' in SWE_chosen)
        
        # Get data from both LOA objects
        SWE_PRI_dic_a = LOA_obj_a.SWE_PRI_dic
        SWE_PRI_dic_b = LOA_obj_b.SWE_PRI_dic
        
        # Extract SWE results for both objects
        SEL_SWE_a = SWE_PRI_dic_a[SWE_chosen]['SWE_results']
        SEL_SWE_b = SWE_PRI_dic_b[SWE_chosen]['SWE_results']
        
        # Filter by Q metric if needed (assuming same Q_chosen for both)
        Q_chosen = SEL_SWE_a['Q_metric'].iloc[0]  # Get Q metric from first object
        SEL_SWE_a = SEL_SWE_a[SEL_SWE_a['Q_metric'] == Q_chosen]
        SEL_SWE_b = SEL_SWE_b[SEL_SWE_b['Q_metric'] == Q_chosen]
        
        # Create category dictionaries for both objects
        SWE_cats_a = {cat: SEL_SWE_a[SEL_SWE_a['category'] == cat] for cat in alphas}
        SWE_cats_b = {cat: SEL_SWE_b[SEL_SWE_b['category'] == cat] for cat in alphas}
        
        # Limit years to common range
        years_a = np.arange(LOA_obj_a.START_YEAR, LOA_obj_a.END_YEAR + 1)
        years_b = np.arange(LOA_obj_b.START_YEAR, LOA_obj_b.END_YEAR + 1)
        common_years = np.intersect1d(years_a, years_b)
        
        for cat in alphas:
            SWE_cats_a[cat] = SWE_cats_a[cat][SWE_cats_a[cat]['year'].isin(common_years)]
            SWE_cats_b[cat] = SWE_cats_b[cat][SWE_cats_b[cat]['year'].isin(common_years)]
        
        # Plot SWE Metrics for first object (LOA_obj_a) - Top subplot
        lowest_SWE_a = 1000 if max1swemetric else 0
        for category, alpha in alphas.items():
            if category in ['LB', 'UB']:
                continue
            subset_a = SWE_cats_a[category]
            if len(subset_a) > 0:
                sns.stripplot(data=subset_a, x='year', y=SWE_chosen, hue='category', ax=ax1,
                            dodge=True, jitter=0.3, alpha=alpha, orient='v', s=markersize[category],
                            palette=colors, zorder=zorder[category],
                            edgecolor='black', linewidth=0.5 if category == 'Posterior' else 0.1)
                
                if category == 'Posterior':
                    val = np.min(subset_a[SWE_chosen]) if max1swemetric else np.max(subset_a[SWE_chosen])
                    if (val < lowest_SWE_a and max1swemetric) or (val > lowest_SWE_a and not max1swemetric):
                        lowest_SWE_a = val
        
        # Plot SWE Metrics for second object (LOA_obj_b) - Bottom subplot
        lowest_SWE_b = 1000 if max1swemetric else 0
        for category, alpha in alphas.items():
            if category in ['LB', 'UB']:
                continue
            subset_b = SWE_cats_b[category]
            if len(subset_b) > 0:
                sns.stripplot(data=subset_b, x='year', y=SWE_chosen, hue='category', ax=ax2,
                            dodge=True, jitter=0.3, alpha=alpha, orient='v', s=markersize[category],
                            palette=colors, zorder=zorder[category],
                            edgecolor='black', linewidth=0.5 if category == 'Posterior' else 0.1)
                
                if category == 'Posterior':
                    val = np.min(subset_b[SWE_chosen]) if max1swemetric else np.max(subset_b[SWE_chosen])
                    if (val < lowest_SWE_b and max1swemetric) or (val > lowest_SWE_b and not max1swemetric):
                        lowest_SWE_b = val
        
        # Calculate global y-limits based on the second axis (bottom plot)
        global_ylim = ((lowest_SWE_b, 1) if max1swemetric else (lowest_SWE_b, 0))
        
        # Set plot properties for top subplot (LOA_obj_a)
        ax1.set_ylim(global_ylim)
        ax1.grid(alpha=0.5)
        ax1.set_title(f'{exp_title_a}', fontsize=16, color='tab:blue', fontweight='bold')
        ax1.legend([], edgecolor='white')  # Hide legend for top plot
        
        # Set plot properties for bottom subplot (LOA_obj_b)
        ax2.set_ylim(global_ylim)
        ax2.grid(alpha=0.5)
        ax2.set_title(f'{exp_title_b}', fontsize=16, color='tab:orange', fontweight='bold')
        
        # Set ylabel based on SWE metric type (same for both subplots)
        if 'grid' in SWE_chosen:
            ylabel = fr"$E_{{GRID}}^{{{translate_SWE_metric_name(SWE_chosen,keep_suffix=False)}}}$ {add_suffix(SWE_chosen)}"
        elif 'elev' in SWE_chosen:
            ylabel = f"ELEV \n {translate_SWE_metric_name(SWE_chosen)} {add_suffix(SWE_chosen)}"
        else:
            ylabel = fr"$E_{{AGG}}^{{{translate_SWE_metric_name(SWE_chosen,keep_suffix=False)}}}$ {add_suffix(SWE_chosen)}"
        
        ax1.set_ylabel(ylabel, fontsize=14)
        ax2.set_ylabel(ylabel, fontsize=14)
        
        # Set x-axis properties
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        ax2.set_xlabel(None)
        ax1.set_xlabel(None)
        
        # Color the axis borders like the original function
        for ax, exp_title in [(ax1, exp_title_a), (ax2, exp_title_b)]:
            for spine in ax.spines.values():
                spine.set_color('tab:blue' if exp_title == 'Fully Synthetic' else 'tab:orange')
                spine.set_linewidth(1.5)
        
        # Create legend for bottom subplot
        legend_elements = []
        legend_labels = []
        
        for category in ['Prior', 'Posterior']:
            if category in alphas:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=colors[category], 
                                                markersize=8, markeredgecolor='black'))
                legend_labels.append(f'{category}')
        
        ax2.legend(legend_elements, legend_labels, loc='best')
        
        # Adjust layout and save
        plt.tight_layout()
        f1.savefig(join(LOA_obj_a.FIGDIR, f"swe_comparison_{SWE_chosen}_{exp_title_a}_vs_{exp_title_b}.png"),
                   bbox_inches='tight', dpi=400)
        plt.show()

    # Main loop
    # for SWE_chosen in ['melt_sum_grid_MAPE']:# ['SWE_melt_NSE_grid']:
    for SWE_chosen in ['melt_sum_APE']:
    # for SWE_chosen in SWE_targets:
        SEL_SWE = SWE_PRI_dic[SWE_chosen]['SWE_results']
        SEL_SWE = SEL_SWE[SEL_SWE['Q_metric'] == Q_chosen]
        SEL_Q = SWE_PRI_dic[SWE_chosen]['Q_results']
        PRI_SWE = SWE_PRI_dic[SWE_chosen]['PRI_SWE']
        PRI_Q = SWE_PRI_dic[SWE_chosen]['PRI_Q']

        SWE_cats = {cat: SEL_SWE[SEL_SWE['category'] == cat] for cat in alphas}
        Q_cats = {cat: SEL_Q[SEL_Q['category'] == cat] for cat in alphas}

        #limit years
        years = np.arange(self.START_YEAR, self.END_YEAR + 1)
        for cat in alphas:
            SWE_cats[cat] = SWE_cats[cat][SWE_cats[cat]['year'].isin(years)]
            Q_cats[cat] = Q_cats[cat][Q_cats[cat]['year'].isin(years)]

        if plot_SWE_metrics:
            plot_swe_metrics(SWE_chosen, SWE_cats, PRI_SWE, alphas, colors, zorder, EXP_ID_translation[self.ORIG_ID])

        if plot_Q_metrics and Q_chosen in ['KGE_meltseason', 'NSE_meltseason']:
            plot_q_metrics(Q_chosen, SWE_chosen, Q_cats, PRI_Q, alphas, colors, dothandles, EXP_ID_translation[self.ORIG_ID])

        if plot_SWE_Q_combo:
            plot_swe_q_combo(SWE_chosen, Q_chosen, SWE_cats, Q_cats, alphas, colors, zorder, dothandles, EXP_ID_translation[self.ORIG_ID])
        if plot_EGU:
            plot_swe_q_combo_EGU(SWE_chosen, Q_chosen, SWE_cats, Q_cats, alphas, colors, zorder, dothandles, EXP_ID_translation[self.ORIG_ID])

    ##%% SWE Comparison between two LOA objects
    # Example usage of the new plot_swe_comparison function
    # Uncomment and modify the following lines to use the function:
    # 
    # # Assuming you have two LOA objects: LOA_obj_a (Syn_244a) and LOA_obj_b (Syn_244b)
    # # and they are stored in a dictionary or list called LOA_objects
    # 
    # # Example call:
    plot_swe_comparison(
        SWE_chosen='melt_sum_APE',
        LOA_obj_a=LOA_objects['Syn_244a'],  # First LOA object
        LOA_obj_b=LOA_objects['Syn_244b'],  # Second LOA object
        alphas=alphas,
        colors=colors,
        zorder=zorder,
        exp_title_a='Fully Synthetic',      # Title for first experiment
        exp_title_b='Semi-Synthetic'        # Title for second experiment
    )

    ##%% Same plots but with Q metrics and SWE metrics 
#%%

    ##%%
    def get_metric_type(metric_name):
        if 'elev' in metric_name:
            return 'Elevation-band average'
        elif 'grid' in metric_name or 'SPAEF' in metric_name:
            return 'Grid average'
        else:
            return 'Catchment-wide'
        
    #Function to translate the SWE metric names to more readable alternatives 
    # def translate_SWE_metric_name(metric_name):
    #     new_name = metric_name
    #     if 'elev' in metric_name:
    #         new_name = metric_name.replace('_elev', '')
    #     elif 'grid' in metric_name:
    #         new_name = metric_name.replace('_grid', '')
            
    #     readable_dic = {
    #         'melt_sum': 'Accumulation',
    #         'SWE_melt_KGE': 'Melt KGE',
    #         'SWE_melt_NSE': 'Melt NSE',
    #         'SWE_NSE': 'SWE NSE',
    #         'SWE_SWS': 'SWS',
    #         't_SWE_start': 'Start of SWE timing',
    #         't_SWE_end': 'End of SWE timing',
    #         't_SWE_max': 'Maximum SWE timing',
    #         'SWE_snowfall_NSE': 'Snowfall NSE',
    #         'SWE_SPAEF': 'Spatial Efficiency'  # Fixed this line
    #     }
    #     for key in readable_dic.keys():
    #         if key in new_name:
    #             # print(key)
    #             new_name = new_name.replace(key, readable_dic[key])
    #     if '_' in new_name:
    #         new_name = new_name.replace('_'+new_name.split('_')[-1], ' error')
    #     # if '_' in new_name:
    #     #     # new_name = new_name.replace('_', ' ')
    #     #     new_name = new_name.split('_')[0]
    #     return new_name


        # if 'elev' in metric_name:
        #     return metric_name.replace('_elev', '')
        # elif 'grid' in metric_name:
        #     return metric_name.replace('_grid', '')
        # else:
        #     return metric_name



    #plot SWE and Q PRI in one figure 
    # SWE_metric_classes = ['catchment','elev','grid']
    # SWE_metric_colormaps = ['Blues','Greens','Reds']
    # SWE_metric_colors = dict(zip(SWE_metric_classes, [sns.color_palette(cmap) for cmap in SWE_metric_colormaps]))
    PRI_SWE_list = []
    PRI_Q_list = []
    for SWE_chosen in SWE_targets:
        PRI_SWE = SWE_PRI_dic[SWE_chosen]['PRI_SWE']
        PRI_SWE['SWE_metric'] = SWE_chosen
        PRI_SWE['SWE_metric_class'] = get_metric_type(SWE_chosen)
        PRI_SWE_list.append(PRI_SWE[PRI_SWE['category'] == 'Posterior'])
        PRI_Q = SWE_PRI_dic[SWE_chosen]['PRI_Q'].copy()
        PRI_Q['SWE_metric'] = SWE_chosen
        PRI_Q['SWE_metric_class'] = get_metric_type(SWE_chosen)
        PRI_Q_list.append(PRI_Q[PRI_Q['category'] == 'UB'])
    PRI_SWE_melted = pd.concat(PRI_SWE_list, axis=0)
    PRI_Q_melted = pd.concat(PRI_Q_list, axis=0)


    # Messy plot of all metrics 
    f1,(ax1,ax2) = plt.subplots(2,1,figsize=(10,12),sharex = True)
    sns.pointplot(data = PRI_SWE_melted, x = 'year',
                    y = 'PRI', hue = 'SWE_metric', ax = ax1,
                    linestyles = '-', palette = 'colorblind')
    sns.pointplot(data = PRI_Q_melted, x = 'year',
                    y = 'PRI', hue = 'SWE_metric', ax = ax2,
                    linestyles = '-', palette = 'colorblind')

    ax1.legend(loc = [1.05, 0.5])
    ax1.set_ylabel('PRI SWE')
    ax1.grid()
    ax2.set_ylabel('PRI Q')
    ax2.grid()
    ax1.axhline(0, color = 'black', linestyle = '--')
    # ax1.set_ylim(top = 1)
    # ax2.set_ylim(top = 1)
    ax2.axhline(0, color = 'black', linestyle = '--')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 45)




    # Create dataframes for plotting with year information
    PRI_SWE_plot = PRI_SWE_melted.copy()
    PRI_Q_plot = PRI_Q_melted.copy()
    PRI_SWE_plot['PRI_type'] = 'SWE PRI'
    PRI_Q_plot['PRI_type'] = 'Q PRI'

    # Combine the dataframes
    yearly_PRI_combined = pd.concat([PRI_SWE_plot, PRI_Q_plot], axis=0)

    # Order SWE metrics by mean SWE PRI values (descending)
    # swe_metric_order = PRI_SWE_melted.groupby('SWE_metric')['PRI'].mean().sort_values(ascending=False).index.tolist()

    ordered_metrics = []
    for metric_type in ['Catchment-wide', 'Elevation-band average', 'Grid average']:
        # Filter metrics of this type
        type_metrics = PRI_SWE_melted[PRI_SWE_melted['SWE_metric_class'] == metric_type]
        
        # If there are metrics of this type, sort them by mean PRI and add to ordered list
        if len(type_metrics) > 0:
            type_order = type_metrics.groupby('SWE_metric')['PRI'].mean().sort_values(ascending=False).index.tolist()
            ordered_metrics.extend(type_order)


    yearly_PRI_combined['SWE_metric'] = pd.Categorical(
        yearly_PRI_combined['SWE_metric'], 
        categories=ordered_metrics, 
        ordered=True
    )



    # Filter only SWE PRI data
    PRI_SWE_only = yearly_PRI_combined[yearly_PRI_combined['PRI_type'] == 'SWE PRI'].copy()
    PRI_SWE_only.to_csv(join(self.LOA_DIR, 'PRI_SWE_only.csv'))
    self.PRI_SWE_only = PRI_SWE_only



# if '244' in EXPS[0]:
#     PR_SWE_only_adir = '/home/pwiersma/scratch/Data/ewatercycle/outputs/Syn_244a/LOA/PRI_SWE_only.csv'
#     PRI_SWE_only_a = pd.read_csv(PR_SWE_only_adir, index_col=0)
#     PRI_SWE_only_bdir = '/home/pwiersma/scratch/Data/ewatercycle/outputs/Syn_244a/LOA_OSHD/PRI_SWE_only.csv'
#     PRI_SWE_only_b = pd.read_csv(PRI_SWE_only_bdir, index_col=0)
# else:
#     PRI_SWE_only_adir = '/home/pwiersma/scratch/Data/ewatercycle/outputs/Syn_162a/LOA/PRI_SWE_only.csv'
#     PRI_SWE_only_a = pd.read_csv(PRI_SWE_only_adir, index_col=0)
#     PRI_SWE_only_bdir = '/home/pwiersma/scratch/Data/ewatercycle/outputs/Syn_162b/LOA/PRI_SWE_only.csv'
#     PRI_SWE_only_b = pd.read_csv(PRI_SWE_only_bdir, index_col=0)
# PRI_SWE_only_a['Setting'] = 'Fully Synthetic'
# PRI_SWE_only_b['Setting'] = 'Semi-Synthetic'
# PRI_SWE_only = pd.concat([PRI_SWE_only_a, PRI_SWE_only_b], axis=0)

#%%
PRI_SWE_only_list = []
for self in LOA_objects.values():
    if 'LB' in self.ORIG_ID:
        continue
    setting = self.EXP_ID_translation[self.ORIG_ID]
    PRI_SWE_only = self.PRI_SWE_only.copy()
    # Add the setting to the dataframe
    PRI_SWE_only['Setting'] = setting
    PRI_SWE_only_list.append(PRI_SWE_only)
PRI_SWE_only = pd.concat(PRI_SWE_only_list, axis=0)


# Add snowfall fraction per year 
f_snow = self.Qsig_obs['Snowfall frac'].copy()
#add f_snow to the PRI_SWE_only dataframe
PRI_SWE_only['f_snow'] = PRI_SWE_only['year'].map(f_snow)

# Define the order of metric types for plotting
# metric_class_order = ['Catchment-wide', 'Elevation-band average', 'Grid average']#
metric_class_order = ['Catchment-wide', 'Grid average']
translated_names = {metric: translate_SWE_metric_name(metric,keep_suffix = False) for metric in PRI_SWE_only['SWE_metric'].unique()}
#%%
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from matplotlib.lines import Line2D

# # — your existing objects assumed in scope —
# # PRI_SWE_only, metric_class_order,
# # translate_SWE_metric_name, translated_names,
# # runid_limit, FIGDIR

# YEAR = 2003
# scenes = ['all']

# # build one Normalize over the full year range
# all_years = PRI_SWE_only['year'].values
# norm = mpl.colors.Normalize(vmin=all_years.min(), vmax=all_years.max())
# vmin,vmax = PRI_SWE_only['f_snow'].min(), PRI_SWE_only['f_snow'].max()
# norm= mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# # two ScalarMappables for the colorbars
# sm_blues   = mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues)
# sm_oranges = mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Oranges)
# for sm in (sm_blues, sm_oranges):
#     sm.set_array([])

# offsets = {'Fully Synthetic': -0.2, 'Semi-Synthetic': 0.2}

# cmaps   = {'Fully Synthetic': plt.cm.Blues,  'Semi-Synthetic': plt.cm.Oranges}

# for f, scene in enumerate(scenes):
#     fig, axes = plt.subplots(
#         nrows=len(metric_class_order), ncols=1,
#         figsize=(6, 9),
#         sharex=(scene != 'Fully catchment')
#     )
#     # *remove all vertical space between subplots*
#     fig.subplots_adjust(hspace=0, top=0.95, bottom=0.12)

#     if len(metric_class_order) == 1:
#         axes = [axes]

#     for i, metric_class in enumerate(metric_class_order):
#         ax = axes[i]

#         # filter
#         df = PRI_SWE_only[PRI_SWE_only['SWE_metric_class'] == metric_class].copy()
#         if scene != 'all':
#             df = df[df['Setting'] == 'Fully Synthetic']

#         # order metrics
#         mets = df['SWE_metric'].unique().tolist()
#         if metric_class == 'Catchment-wide' and mets:
#             print('hhaa')
#             order = (
#                 df[df['Setting']=='Fully Synthetic']
#                   .groupby('SWE_metric')['PRI']
#                   .mean().sort_values().index.tolist()
#             )
#             order = [m for m in order if m in mets]
#         else:
#             order = order
#         print(order)

#         # translate & categorize
#         df['SWE_metric_translated'] = df['SWE_metric'].map(translated_names)
#         translated_order = [
#             translate_SWE_metric_name(m, keep_suffix=False)
#             for m in order
#         ]
#         df['SWE_metric_translated'] = pd.Categorical(
#             df['SWE_metric_translated'],
#             categories=translated_order,
#             ordered=True
#         )

#         # 1) slight y-jitter + colormap scatter
#         for setting, cmap in cmaps.items():
#             sub = df[df['Setting']==setting]
#             if sub.empty: 
#                 continue

#             y0 = sub['SWE_metric_translated'].cat.codes + offsets[setting]
#             # tiny jitter around that offset:
#             y0 = y0 + np.random.uniform(-0.1, 0.1, size=len(y0))

#             yrs = sub['f_snow'].values
#             cols = cmap(norm(yrs))

#             ax.scatter(
#                 sub['PRI'], y0,
#                 c=cols,
#                 edgecolors='black', linewidths=0.5,
#                 alpha=0.7, s=50
#             )

#         # 2) diamonds + error bars via seaborn.pointplot
#         sns.pointplot(
#             data=df,
#             x='PRI',
#             y='SWE_metric_translated',
#             hue='Setting',
#             order=translated_order,
#             dodge=0.4,
#             join=True,
#             capsize=0,
#             errorbar = 'ci',
#             errwidth=2,
#             linewidth = 2,
#             markers='D',
#             markersize = 7,
#             palette={'Fully Synthetic':'tab:blue','Semi-Synthetic':'tab:orange'},
#             ax=ax
#         )
#         # remove the per-subplot seaborn legend it creates
#         for child in ax.get_children():
#             if isinstance(child, mpl.legend.Legend):
#                 child.remove()

#         # vertical reference lines
#         ax.axvline((self.runid_limit+1)/2,   color='black', linestyle='--', alpha=0.7)
#         ax.axvline((self.runid_limit/100+1)/2, color='black', linestyle='--', alpha=0.7)

#         # y-axis labels & invert
#         ax.set_yticks(range(len(translated_order)))
#         ax.set_yticklabels(translated_order, fontsize=9)
#         # ax.invert_yaxis()
#         ax.set_xlim(0, self.runid_limit/2 + 400)
#         ax.grid(alpha=0.3)

#         # ylabel per class
#         label_map = {
#             'Catchment-wide':         'Catchment-wide error\n(AGG)',
#             'Grid average':           'Grid-averaged error\n(GRID)',
#             'Elevation-band average': 'Elevation-averaged error\n(ELEV)'
#         }
#         ax.set_ylabel(label_map.get(metric_class, metric_class), fontsize=10)

#     # common xlabel
#     # fig.text(0.45, 0.05, 'Median rank of 50 posterior runs', ha='center', fontsize=11)
#     axes[-1].set_xlabel('Median rank of 50 posterior runs', fontsize=11)          
    
#     axes[-1].text(
#                 0.025, -0.05, 'Perfect Q\nconstraint',
#                 ha='center', va='top',
#                 transform=axes[-1].transAxes,
#                 fontsize=10, rotation=30, fontstyle='italic'
#             )
#     axes[-1].text(
#                 0.825, -0.05, 'No Q\nconstraint',
#                 ha='center', va='top',
#                 transform=axes[-1].transAxes,
#                 fontsize=10, rotation=30, fontstyle='italic'
#             )
#     # extend xticks to include 25
#     all_xt = np.append(axes[-1].get_xticks(), 25)
#     axes[-1].set_xticks(np.unique(all_xt)[1:])
#     # legend for round points only
#     legend_handles = [
#         Line2D([0],[0], marker='o', color='tab:blue',  alpha=0.6, linestyle='', markersize=6),
#         Line2D([0],[0], marker='o', color='tab:orange',alpha=0.6, linestyle='', markersize=6),
#     ]
#     legend_labels = ['Fully Synthetic years','Semi-Synthetic years']
#     axes[0].legend(legend_handles, legend_labels, loc='upper right', fontsize=9)

#     # leave room for 2 colorbars on the right
#     # plt.tight_layout(rect=[0,0.1,0.87,1])
#     cax1 = fig.add_axes([0.905+0.01, 0.15, 0.02, 0.7])
#     cax2 = fig.add_axes([0.93+0.01,0.15, 0.02, 0.7])
#     cb1  = fig.colorbar(sm_blues,   cax=cax1)
#     cb2  = fig.colorbar(sm_oranges, cax=cax2)

#     # only lowest & highest ticks, as integers
#     cb2.set_ticks([norm.vmin, norm.vmax])
#     # cb2.set_ticklabels([int(norm.vmin), int(norm.vmax)])
#     cb1.ax.yaxis.set_ticks([])

#     cb2.ax.set_ylabel('Snowfall fraction [-]', rotation=90, labelpad=-6)

#     plt.savefig(f"{self.FIGDIR}/yearly_SWE_PRI_by_class_{f}.svg", dpi=300, bbox_inches='tight')
#     plt.show()



#%%
# Create a figure with three subplots, one for each metric type
YEAR = 2003

figures = []
# scenes = ['Fully catchment','Fully all','all']
scenes = ['all']

for f in range(len(scenes)):
   
    fig, axes = plt.subplots(len(metric_class_order), 1, figsize=(6, 9), 
                sharex=True if scenes[f]!='Fully catchment' else False)
    plt.subplots_adjust(hspace=0.0)

    # Process each metric class in its own subplot
    for i, metric_class in enumerate(metric_class_order):
        if scenes[f] == 'Fully catchment':
            for axes in axes[1:]:
                axes.remove()
                continue
        # Create a new figure for each metric class
        
        # Filter data for this metric class
        class_data = PRI_SWE_only[PRI_SWE_only['SWE_metric_class'] == metric_class]
        
        # Get metrics in this class ordered by mean PRI value
        metrics_in_class = list(class_data['SWE_metric'].unique())

        # Sort metrics in this class by mean PRI value
        if metric_class == 'Catchment-wide':
            if len(metrics_in_class) > 0:
                order = class_data[class_data['Setting'] == 'Fully Synthetic'].groupby('SWE_metric')['PRI'].mean().sort_values(ascending=True).index.tolist()
                #remove any classes thare not in the metrics_in_class
                order = [metric for metric in order if metric in metrics_in_class]
            else:
                order = metrics_in_class
            # Create categorical variable for ordered plotting
            class_data['SWE_metric'] = pd.Categorical(
                class_data['SWE_metric'], 
                categories=order,
                ordered=True
            )
            # print(order)
            
            translated_order = [translate_SWE_metric_name(metric,keep_suffix=False) for metric in order]
        # translated_order = [translate_SWE_metric_name(metric) for metric in metrics_in_class]

        # if metric_class == 'Grid average':
        #     # print('Grid average')
        #     translated_order.append('Spatial Efficiency') 

        class_data['SWE_metric_translated'] = class_data['SWE_metric'].map(translated_names)
        # print(class_data['SWE_metric_translated'])
        class_data['SWE_metric_translated'] = pd.Categorical(
            class_data['SWE_metric_translated'], 
            categories=translated_order,
            ordered=True
        ) 
        if scenes[f] != 'all':
            class_data = class_data[class_data['Setting'] == 'Fully Synthetic']
     
        # Create stripplot for this metric class
        # sns.stripplot(
        #     data=class_data, 
        #     x='PRI', 
        #     y='SWE_metric_translated',
        #     ax=axes[i],
        #     # color='tab:blue',
        #     hue = 'Setting',
        #     alpha=0.2,
        #     jitter=0.25,
        #     size=8,
        #     dodge=True,
        #     edgecolor='black',
        #     linewidth=0.5,
        # )
        # colors_dic = {'Fully Synthetic': 'Blues','Semi-Synthetic': 'Oranges'}
        # y = [np.arange(5) + i*0.1 for i in range(len(class_data['Setting'].unique()))]
        # class_data['Y0'] = class_data['SWE_metric_translated'].cat.codes + 0.1 * class_data['Setting'].cat.codes

        # for s,set in enumerate(class_data['Setting'].unique()):
        #     sns.stripplot(
        #         data=class_data[class_data['Setting'] == set], 
        #         x='PRI', 
        #         y='Y0',#'SWE_metric_translated',
        #         ax=axes[i],
        #         # color='tab:blue',
        #         hue = 'year',
        #         palette = colors_dic[set],
        #         alpha=0.5,
        #         jitter=0,
        #         size=8,
        #         dodge=s+0.1,
        #         edgecolor='black',
        #         linewidth=0.5,
        # )
        #make one stripplot for all years except YEAR, and one just for YEAR 
        # if (f==0) & (i ==1):
        #     pointalpha = 0
        # else:
        pointalpha = 0.3
        sns.stripplot(
            data=class_data[class_data['year'] != YEAR], 
            x='PRI', 
            y='SWE_metric_translated',
            ax=axes[i],
            hue='Setting',
            alpha=pointalpha,
            jitter=0.25,
            size=8,
            dodge=True,
            marker = 'o'
        )
        sns.stripplot(  
            data=class_data[class_data['year'] == YEAR], 
            x='PRI', 
            y='SWE_metric_translated',
            ax=axes[i],
            hue='Setting',
            alpha=pointalpha,
            jitter=0.25,
            size=8,
            dodge=True,
            marker = 'o',
            edgecolor = 'black',
            linewidth = 2
        )
        
        sns.pointplot(
            data=class_data,
            x='PRI',
            y='SWE_metric_translated',
            # errorbar = ('pi',50),
            errorbar = 'ci',
            # capsize = 0.1,
            ax=axes[i],
            hue='Setting',
            marker='D',
            markersize=7,
            dodge = 0.15 if scenes[f] =='all' else False,
            alpha = 0 if (scenes[f]=='Fully catchment') & (i ==1) else 1,
            zorder = 100,
        )
     
        # Add vertical line at 0
        # axes[i].axvline(0, color='black', linestyle='--', alpha=0.7)
        Nprior = self.runid_limit
        axes[i].axvline((Nprior+1)/2 , color='black', linestyle='--', alpha=0.7)
        Nposterior = self.runid_limit/100
        axes[i].axvline((Nposterior+1)/2, color='black', linestyle='--', alpha=0.7)

        #annotate at the Nprior and Nposterior lines below the x-axis
        if i ==len(metric_class_order)-1:   
            axes[i].text((Nprior+1)/2, 4.7, 'No Q \n constraint', ha='center',
                        va='top', fontsize=10, rotation = 30,fontstyle='italic')
            axes[i].text((Nposterior+1)/2, 4.7, 'Perfect Q \n constraint',fontstyle = 'italic',
                        ha='center', va='top', fontsize=10, rotation = 30)

            xticks = axes[i].get_xticks()
            newticks = np.array([int(x) for x in xticks]+[25])
            print(newticks)
            axes[i].set_xticks(newticks)

        # Customize the subplot
        axes[i].grid(alpha=0.3)
        # axes[i].set_title(f'{metric_class.capitalize()} Metrics', fontsize=12)
        if metric_class =='Catchment-wide':
            # metric_class_label = 'Catchment-wide (AGG)'
            # metric_class_label = 'Catchment-wide error \n (AGG)'
            # metric_class_label = 'Catchment-wide error \n '+r'$E_{AGG}$'
            metric_class_label = r"$E_{AGG}$"
        elif metric_class == 'Grid average':
            # metric_class_label = 'Grid average (GRID)'
            # metric_class_label = 'Grid-averaged error \n (GRID)'
            # metric_class_label = 'Grid-averaged error \n '+r'$E_{GRID}$'
            metric_class_label = r"$E_{GRID}$"
        elif metric_class == 'Elevation-band average':
            # metric_class_label = 'Elevation-band average (ELEV)'
            metric_class_label = 'Elevation-averaged error \n (ELEV)'
        axes[i].set_ylabel(metric_class_label, fontsize=12)
        axes[i].set_xlim(left = 0)
        axes[i].legend([],edgecolor ='white')
        axes[i].set_xlabel(None)
    # Set the common x-label
    # fig.text(0.5, 0.04, 'Posterior Rank Improvement (PRI)', ha='center', fontsize=12)
    # fig.text(0.45, 0.07, 'Median rank of 50 posterior runs', ha='center', fontsize=11)
    fig.text(0.45, 0.06, r'$R_{post,median}$', ha='center', fontsize=14)
    #add an xtick at 15 with label


    # Create custom legend
    from matplotlib.lines import Line2D
    # custom_handles = [
    #     Line2D([0], [0], marker='o', color='tab:blue', markersize=8, alpha=0.7, linestyle=''),
    #     Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:blue', markeredgecolor='black', markersize=10)
    # ]
    custom_handles = [
        Line2D([0], [0], marker='o', color='tab:blue', markersize=8, alpha=0.4, linestyle=''),
        Line2D([0], [0], marker='o',color= 'tab:orange', markersize=8, alpha=0.4, linestyle=''),
    ]
    custom_labels = ['Fully Synthetic', 'Semi-Synthetic']
    axes[0].legend(custom_handles, custom_labels, loc='upper right')#, bbox_to_anchor=(1.05, 1.0))
    axes[0].set_xlim(left = 1,right = self.runid_limit/2 + 400)
    # Add overall title
    # plt.suptitle('SWE PRI Values by Metric Class and Type', fontsize=14, y=0.98)

    # Adjust layout
    # plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
    # plt.savefig(join(self.FIGDIR, f"yearly_SWE_PRI_by_class_{f}.svg"), dpi=300, bbox_inches='tight')
    # figures.append(fig)
    plt.savefig(join("/home/pwiersma/scratch/Figures/Paper1", f"Fig4_E_rpostmedian.svg"), bbox_inches='tight')

for fig in figures:
    plt.show()
 
#%% FOR POSTER
# YEAR = 2014

# figures = []
# # scenes = ['Fully catchment','Fully all','all']
# scenes = ['all']

# for f in range(len(scenes)):
   
#     fig, axes = plt.subplots(len(metric_class_order), 1, figsize=(6,8), 
#                 sharex=True if scenes[f]!='Fully catchment' else False)
#     plt.subplots_adjust(hspace=0.0)

#     # Process each metric class in its own subplot
#     for i, metric_class in enumerate(metric_class_order):
#         if scenes[f] == 'Fully catchment':
#             for axes in axes[1:]:
#                 axes.remove()
#                 continue
#         # Create a new figure for each metric class
        
#         # Filter data for this metric class
#         class_data = PRI_SWE_only[PRI_SWE_only['SWE_metric_class'] == metric_class]
        
#         # Get metrics in this class ordered by mean PRI value
#         metrics_in_class = list(class_data['SWE_metric'].unique())

#         # Sort metrics in this class by mean PRI value
#         if metric_class == 'Catchment-wide':
#             if len(metrics_in_class) > 0:
#                 order = class_data[class_data['Setting'] == 'Fully Synthetic'].groupby('SWE_metric')['PRI'].mean().sort_values(ascending=True).index.tolist()
#                 #remove any classes thare not in the metrics_in_class
#                 order = [metric for metric in order if metric in metrics_in_class]
#             else:
#                 order = metrics_in_class
#             # Create categorical variable for ordered plotting
#             class_data['SWE_metric'] = pd.Categorical(
#                 class_data['SWE_metric'], 
#                 categories=order,
#                 ordered=True
#             )
#             # print(order)
            
#             translated_order = [translate_SWE_metric_name(metric,keep_suffix= False) for metric in order]
#         # translated_order = [translate_SWE_metric_name(metric) for metric in metrics_in_class]

#         # if metric_class == 'Grid average':
#         #     # print('Grid average')
#         #     translated_order.append('Spatial Efficiency') 

#         class_data['SWE_metric_translated'] = class_data['SWE_metric'].map(translated_names)
#         # print(class_data['SWE_metric_translated'])
#         class_data['SWE_metric_translated'] = pd.Categorical(
#             class_data['SWE_metric_translated'], 
#             categories=translated_order,
#             ordered=True
#         ) 
#         if scenes[f] != 'all':
#             class_data = class_data[class_data['Setting'] == 'Fully Synthetic']
     
#         # Create stripplot for this metric class
#         sns.stripplot(
#             data=class_data, 
#             x='PRI', 
#             y='SWE_metric_translated',
#             ax=axes[i],
#             # color='tab:blue',
#             hue = 'Setting',
#             alpha=0.25,
#             jitter=0.25,
#             size=8,
#             dodge=True,
#             edgecolor='black',
#             linewidth=0.5,
#         )
#         #make one stripplot for all years except YEAR, and one just for YEAR 
#           # if (f==0) & (i ==1):
#         #     pointalpha = 0
#         # else:
#         # pointalpha = 0.3
#         # sns.stripplot(
#         #     data=class_data[class_data['year'] != YEAR], 
#         #     x='PRI', 
#         #     y='SWE_metric_translated',
#         #     ax=axes[i],
#         #     hue='Setting',
#         #     alpha=pointalpha,
#         #     jitter=0.25,
#         #     size=8,
#         #     dodge=True,
#         #     marker = 'o'
#         # )
#         # sns.stripplot(  
#         #     data=class_data[class_data['year'] == YEAR], 
#         #     x='PRI', 
#         #     y='SWE_metric_translated',
#         #     ax=axes[i],
#         #     hue='Setting',
#         #     alpha=pointalpha,
#         #     jitter=0.25,
#         #     size=8,
#         #     dodge=True,
#         #     marker = 'o',
#         #     edgecolor = 'black',
#         #     linewidth = 2
#         # )
        
#         sns.pointplot(
#             data=class_data,
#             x='PRI',
#             y='SWE_metric_translated',
#             # errorbar = ('pi',50),
#             errorbar = 'ci',
#             # capsize = 0.1,
#             ax=axes[i],
#             hue='Setting',
#             marker='D',
#             markersize=7,
#             dodge = 0.3 if scenes[f] =='all' else False,
#             alpha = 0 if (scenes[f]=='Fully catchment') & (i ==1) else 1,
#             zorder = 100,
#         )
     
#         # Add vertical line at 0
#         # axes[i].axvline(0, color='black', linestyle='--', alpha=0.7)
#         Nprior = self.runid_limit
#         axes[i].axvline((Nprior+1)/2 , color='black', linestyle='--', alpha=0.7)
#         Nposterior = self.runid_limit/100
#         axes[i].axvline((Nposterior+1)/2, color='black', linestyle='--', alpha=0.7)

#         #annotate at the Nprior and Nposterior lines below the x-axis
#         if i ==len(metric_class_order)-1:   
#             axes[i].text((Nprior+1)/2, 4.7, 'No Q \n constraint', ha='center',
#                         va='top', fontsize=11, rotation = 30,fontstyle='italic')
#             axes[i].text((Nposterior+1)/2, 4.7, 'Perfect Q \n constraint',fontstyle = 'italic',
#                         ha='center', va='top', fontsize=11, rotation = 30)

#         # Customize the subplot
#         axes[i].grid(alpha=0.3)
#         # axes[i].set_title(f'{metric_class.capitalize()} Metrics', fontsize=12)
#         if metric_class =='Catchment-wide':
#             metric_class_label = 'Catchment-wide error'
#         elif metric_class == 'Grid average':
#             metric_class_label = 'Grid averaged error'
#         elif metric_class == 'Elevation-band average':
#             metric_class_label = 'Elevation-band average (ELEV)'
#         axes[i].set_ylabel(metric_class_label, fontsize=10)
#         axes[i].set_xlim(left = 0)
#         axes[i].legend([],edgecolor ='white')
#         axes[i].set_xlabel(None)
#     # Set the common x-label
#     # fig.text(0.5, 0.04, 'Posterior Rank Improvement (PRI)', ha='center', fontsize=12)
#     # fig.text(0.5, 0.04, 'Annual median Posterior rank \n among 5000 Prior runs', ha='center', fontsize=12)
#     fig.text(0.46,0.04,'Median rank of 50 posterior runs \n (2001-2022)',
#              ha='center', fontsize=11)

#     # Create custom legend
#     from matplotlib.lines import Line2D
#     # custom_handles = [
#     #     Line2D([0], [0], marker='o', color='tab:blue', markersize=8, alpha=0.7, linestyle=''),
#     #     Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:blue', markeredgecolor='black', markersize=10)
#     # ]
#     custom_handles = [
#         Line2D([0], [0], marker='o', color='tab:blue', markersize=8, alpha=0.4, linestyle=''),
#         Line2D([0], [0], marker='o',color= 'tab:orange', markersize=8, alpha=0.4, linestyle=''),
#         # Line2D([0], [0], marker='D', color='w', markerfacecolor='white', markeredgecolor='white)', markersize=0),
#     ]
#     # custom_labels = ['Fully Synthetic', 'Semi-Synthetic','(One point for each year)']
#     custom_labels = ['Fully Synthetic', 'Semi-Synthetic']

#     axes[0].legend(custom_handles, 
#                    custom_labels, loc='upper right',
#                    fontsize = 11,
#                    title = 'One point for each year:',
#                    title_fontsize = 11)#, bbox_to_anchor=(1.05, 1.0))
#     axes[0].set_xlim(left = 1,right = self.runid_limit/2 + 400)
#     # Add overall title
#     # plt.suptitle('SWE PRI Values by Metric Class and Type', fontsize=14, y=0.98)

#     # Adjust layout
#     # plt.tight_layout(rect=[0, 0.05, 0.85, 0.95])
#     plt.savefig(join(self.FIGDIR, f"yearly_SWE_PRI_by_class_POSTER_{f}.svg"), dpi=500, bbox_inches='tight')
#     figures.append(fig)

# for fig in figures:
#     plt.show()
#%%

# Create the Figure for both Q PRI and SWE PRI 
# plt.figure(figsize=(12, 10))

# # Create stripplot with jittered points
# ax = sns.stripplot(
#     data=yearly_PRI_combined, 
#     x='PRI', 
#     y='SWE_metric',
#     hue='PRI_type',
#     palette=['tab:blue', 'tab:orange'],
#     dodge=True,  # Separate SWE and Q points horizontally
#     alpha=0.7,
#     jitter=0.25,
#     size=8
# )

# # Add mean values as markers with black edges
# for pri_type, color in zip(['SWE PRI', 'Q PRI'], ['tab:blue', 'tab:orange']):
#     means = yearly_PRI_combined[yearly_PRI_combined['PRI_type'] == pri_type].groupby('SWE_metric')['PRI'].mean()
#     for metric in means.index:
#         plt.scatter(
#             means[metric], 
#             metric, 
#             marker='D',  # Diamond shape
#             s=120,
#             color=color,
#             edgecolor='black',
#             linewidth=1.5,
#             zorder=10
#         )

# # Add vertical line at 0
# plt.axvline(0, color='black', linestyle='--', alpha=0.7)

# # Customize the plot
# plt.grid(alpha=0.3)
# plt.title('PRI Values by SWE Metric (Individual Years)', fontsize=14)
# plt.xlabel('Posterior Rank Improvement (PRI)', fontsize=12)
# plt.ylabel('SWE Metric', fontsize=12)

# # Legend with custom labels
# handles, labels = ax.get_legend_handles_labels()
# custom_handles = handles + [
#     plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:blue', 
#                markeredgecolor='black', markersize=10),
#     plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:orange',
#                markeredgecolor='black', markersize=10)
# ]
# custom_labels = labels + ['SWE PRI Mean', 'Q PRI Mean']
# plt.legend(custom_handles, custom_labels, title='PRI Type', loc='upper left', bbox_to_anchor=(1.05, 1))

# # Adjust layout
# plt.tight_layout()
# plt.savefig('yearly_PRI_by_metric.png', dpi=300, bbox_inches='tight')
# plt.show()
#%%
for self in LOA_objects.values():
    if 'LB' in self.ORIG_ID:
        continue
    for year in [YEAR]:
        # target_SWE_metric = 'SWE_melt_NSE_grid'
        target_SWE_metric = 'melt_sum_grid_MAPE'
        SEL = SWE_PRI_dic[SWE_chosen]['SWE_results'][SWE_PRI_dic[SWE_chosen]['SWE_results']['Q_metric'] == Q_chosen]
        all_ids = SEL[SEL['year'] == year]['index']
        post_ids = SEL[(SEL['category'] == 'Posterior') & (SEL['year'] == year)]['index'].values
        ub_ids = SWE_PRI_dic[target_SWE_metric]['Q_UB'][YEAR].columns

        #SWE plot
        SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
        SWE_sims_all = self.SWE[year]
        SWE_sims_post = SWE_sims_all[post_ids]
        SWE_ub = SWE_sims_all[ub_ids]

        # obs_metric = self.calc_melt_sum_grid(SWEobs)
        # ape_values = []
        # swe_subset = SWE_sims_all[closest_parset.index]
        # for idx in swe_subset.data_vars:
        #     swesim = swe_subset[idx]
        #     swesim_metric = self.calc_melt_sum_grid(swesim)
        #         # Calculate APE
        #     APE = (swesim_metric - obs_metric) / obs_metric * 100
        #     APE = APE.where(APE.notnull(), np.nan)

        #     plt.figure()
        #     APE.plot()
            
        #     #calc APE between the two and plot the grid
        #     ape_value = APE.isel(lat=9, lon=2).values
        #     ape_values.append(ape_value)
        # ape_values = np.array(ape_values)
        # plt.figure(figsize=(10, 6))
        # plt.plot(ape_values, label='APE', color='tab:red',linewidth = 0,
        #          marker='o', markersize=3, alpha=0.5)

        SWEobs_bands = self.E.swe2bands(SWEobs,bands = list(self.elev_bands))
        SWE_sims_all_bands = self.E.swe2bands(SWE_sims_all,bands = list(self.elev_bands))
        SWEsims_good_bands = self.E.swe2bands(SWE_sims_post,bands = list(self.elev_bands))
        SWE_ub_bands = self.E.swe2bands(SWE_ub,bands = list(self.elev_bands))
        bands  = list(SWEobs_bands.keys())

        #find closest parameter sets: favourites 
        if self.EXP_ID_translation[self.EXP_ID]== 'Fully Synthetic':
            parsets = self.pars[year].copy()
            truepars = self.trueparams.copy()
            truepars['sfcf'] = truepars[f"sfcf_{year}"]
            param_ranges = self.param_ranges.copy()
            #normalize parameters
            for col in parsets.columns:
                parsets[col] = (parsets[col] - param_ranges[col][0]) / (param_ranges[col][1] - param_ranges[col][0])
                truepars[col] = (truepars[col] - param_ranges[col][0]) / (param_ranges[col][1] - param_ranges[col][0])
            #find closest parameter sets
            for col in parsets.columns:
                parsets[col] = np.abs((parsets[col] - truepars[col]))
            # parsets = parsets[['sfcf','sfcf_scale','tt_scale','TT']]
            parsets['distance'] = parsets.sum(axis=1)
            closest_parset = parsets.nsmallest(10, 'distance').index
            closest_parset = parsets.loc[closest_parset]

            SWE_fav = SWE_sims_all[closest_parset.index]
            SWE_fav_bands = self.E.swe2bands(SWE_fav, bands=list(self.elev_bands))
            # print(SWE_fav_bands.keys())

        axi, axj = 2,int(np.ceil(len(bands)/2))
        f1, axes = plt.subplots(axi,axj, figsize=(2 * len(bands), 8))
        axes = axes.flatten()
        for i, band in enumerate(bands):
            ax1 = axes[i]
            obsplot = SWEobs_bands[band].plot(ax=ax1, color='black', 
                                                label='SWEobs', zorder=100,
                                                linestyle = '--')
            SWE_all_t = SWE_sims_all_bands[band].drop(columns='spatial_ref', errors='ignore')
            goodsims = SWEsims_good_bands[band].drop(columns='spatial_ref', errors='ignore')
            simplot = goodsims.plot(ax=ax1, color=colors['Posterior'], label='_noLegend', legend=False,
                                     linestyle='-', alpha = 0.5)
            SWE_ub_band = SWE_ub_bands[band].drop(columns='spatial_ref', errors='ignore')
            # ax1.plot(SWE_ub_band.index, SWE_ub_band, color=colors['UB'], linestyle='-', label='Upper Benchmark',
            #          alpha  = 0.5, zorder=50)
            # if self.EXP_ID_translation[self.EXP_ID] == 'Fully Synthetic':
            #     SWE_fav_band = SWE_fav_bands[band].drop(columns='spatial_ref', errors='ignore')
            #     ax1.plot(SWE_fav_band.index, SWE_fav_band, color='tab:orange', linestyle='-', label='Fav. Parameter Set',
            #              alpha  = 0.5, zorder=50)
            # ax1.legend(handles=linehandles, labels=['SWEobs', 'SWEsim_selection', 'All SWEsim', 'Upper benchmark', 'Lower benchmark'])
            ax1.legend([])
            ax1.set_title(f"{band}")
            ax1.grid()
        # plt.show()

        # handles = [Line2D([0], [0], color='black', linestyle='--')]
        handles = [linehandle_obs,linehandles['Posterior']]
        labels = ['Observed', 'Posterior']
        ax1.legend(handles, labels)
        ax1.set_title(f"{band}")
        plt.suptitle(f"SWE for {year} by Elevation Band \n SWE target: {target_SWE_metric} ", fontsize=16)
        plt.show()

        #melt per band
        def swe_bands_to_melt(swe_bands):
            diff = swe_bands.diff()
            melt_bands = diff.where(diff < 0, 0) * -1
            return melt_bands
        axi, axj = 2,int(np.ceil(len(bands)/2))
        f1, axes = plt.subplots(axi,axj, figsize=(2 * len(bands), 8))
        axes = axes.flatten()
        for i, band in enumerate(bands):
            ax1 = axes[i]
            obsplot = swe_bands_to_melt(SWEobs_bands[band]).plot(ax=ax1, color='black', 
                                                label='SWEobs', zorder=100,
                                                linestyle = '--')
            SWE_all_t = swe_bands_to_melt(SWE_sims_all_bands[band]).drop(columns='spatial_ref', errors='ignore')
            ax1.fill_between(SWE_all_t.index, SWE_all_t.min(axis=1), SWE_all_t.max(axis=1),
                            color=colors['Prior'], alpha=0.2, zorder=0)
            goodsims = swe_bands_to_melt(SWEsims_good_bands[band]).drop(columns='spatial_ref', errors='ignore')
            simplot = goodsims.plot(ax=ax1, color=colors['Posterior'], label='_noLegend', legend=False,
                                     linestyle='-', alpha = 0.5)
            SWE_ub_band = swe_bands_to_melt(SWE_ub_bands[band]).drop(columns='spatial_ref', errors='ignore')
            # ax1.plot(SWE_ub_band.index, SWE_ub_band, color=colors['UB'], linestyle='-', label='Upper Benchmark',
            #          alpha  = 0.5, zorder=50)
            # if self.EXP_ID_translation[self.EXP_ID] == 'Fully Synthetic':
            #     SWE_fav_band = swe_bands_to_melt(SWE_fav_bands[band]).drop(columns='spatial_ref', errors='ignore')
            #     ax1.plot(SWE_fav_band.index, SWE_fav_band, color='tab:orange', linestyle='-', label='Fav. Parameter Set',
            #              alpha  = 0.5, zorder=50)
            # ax1.legend(handles=linehandles, labels=['SWEobs', 'SWEsim_selection', 'All SWEsim', 'Upper benchmark', 'Lower benchmark'])
            ax1.legend([])
            ax1.set_title(f"{band}")
            ax1.set_xlim(pd.Timestamp(f"{year}-03-01"), pd.Timestamp(f"{year}-06-30"))
            ax1.set_ylim(0,80)
            ax1.grid()
        # plt.show()

        # handles = [Line2D([0], [0], color='black', linestyle='--')]
        handles = [linehandle_obs,linehandles['Posterior']]
        labels = ['Observed', 'Posterior']
        ax1.legend(handles, labels)
        ax1.set_title(f"{band}")
        plt.suptitle(f"SWE for {year} by Elevation Band \n SWE target: {target_SWE_metric} ", fontsize=16)
        plt.show()


        #make catchment-wide plot 
        SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
        SWE_sims_all = self.SWE[year]
        SWE_sims_post = SWE_sims_all[post_ids]
        SWE_sims_UB = SWE_sims_all[ub_ids]
        SWEobs_catchment = self.calc_sum2d(SWEobs).to_pandas()
        SWE_sims_all_catchment = self.calc_sum2d(SWE_sims_all).to_pandas()
        SWEsims_good_catchment = self.calc_sum2d(SWE_sims_post).to_pandas()  # Added to_pandas() here
        SWE_sims_UB_catchment = self.calc_sum2d(SWE_sims_UB).to_pandas()
        f1,ax1 = plt.subplots(1,1,figsize=(10,6))
        obsplot = SWEobs_catchment.plot(ax=ax1, color='black', 
                                        label='SWEobs', zorder=100,
                                        linestyle = '--')
        SWE_all_t = SWE_sims_all_catchment.drop(columns='spatial_ref', errors='ignore')
        goodsims = SWEsims_good_catchment.drop(columns='spatial_ref', errors='ignore')
        simplot = goodsims.plot(ax=ax1, color=colors['Posterior'], label='_noLegend', legend=False,
                                alpha=alphas['Posterior'], linestyle='-')
        ax1.fill_between(SWE_all_t.index, SWE_all_t.min(axis=1), SWE_all_t.max(axis=1),
                        color=colors['Prior'], alpha=0.2, zorder=0)
        labels = ['SWEobs', 'Posterior', 'Prior']
        handles = [linehandle_obs, linehandles['Posterior'], patch_handles['Prior']]
        ax1.legend(handles, labels)
        ax1.set_title(f"Catchment-wide SWE \n {self.EXP_ID_translation[self.EXP_ID]} {year}", fontsize=16)
        ax1.grid()
        plt.show()

        #make melt plot
        def swe3d_to_melt(swe3d):
                swe1d = swe3d.sum(dim = ['lat','lon'])
                swe3d_melt = swe1d.diff('time')
                swe3d_melt = xr.where(swe3d_melt > 0, 0, swe3d_melt)*-1
                melt1d = swe3d_melt.to_pandas()
                if isinstance(melt1d, pd.DataFrame) and 'spatial_ref' in melt1d.columns:
                    melt1d = melt1d.drop(columns = 'spatial_ref')
                return melt1d
        def swe3d_to_snowfall(swe3d):
            swe1d = swe3d.sum(dim = ['lat','lon'])
            swe3d_snowfall = swe1d.diff('time')
            swe3d_snowfall = xr.where(swe3d_snowfall < 0, 0, swe3d_snowfall)
            snowfall1d = swe3d_snowfall.to_pandas()
            if isinstance(snowfall1d, pd.DataFrame) and 'spatial_ref' in snowfall1d.columns:
                snowfall1d = snowfall1d.drop(columns = 'spatial_ref')
            return snowfall1d
        
        SWEobs_melt = swe3d_to_melt(SWEobs)
        SWE_sims_all_melt = swe3d_to_melt(SWE_sims_all)
        SWE_sims_post_melt = swe3d_to_melt(SWE_sims_post)
        SWE_sims_UB_melt = swe3d_to_melt(SWE_sims_UB)
        # SWE_sims_LB_melt = swe3d_to_melt(SWE_sims_LB)


        f1,ax1 = plt.subplots(1,1,figsize=(10,6))
        handles = [dothandle_obs, linehandle_obs, linehandles['Posterior'], linehandles['UB'], patchhandle_obs]
        labels = ['SWEobs', 'Posterior', 'Upper benchmark', 'Lower benchmark']
        # Plot total melt
        SWEobs_melt.plot(ax=ax1, color='black', label='SWEobs', linestyle='--',zorder=102)
        SWE_sims_post_melt.plot(ax=ax1, color=colors['Posterior'], 
                                    label='_noLegend', legend=None,
                                alpha = 0.7)
        ax1.fill_between(SWE_sims_all_melt.index, 
                            SWE_sims_all_melt.min(axis=1), 
                            SWE_sims_all_melt.max(axis=1), 
                            color='tab:blue', alpha=0.2)
        # SWE_sims_UB_melt.plot(ax=ax1, color=colors['UB'], linestyle='--', label='Upper benchmark',zorder = 1)
        # ax1.fill_between(SWE_sims_all_melt.index, SWE_sims_LB_melt.min(axis=1), SWE_sims_LB_melt.max(axis=1), color=colors['LB'], alpha=0.2)
        ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
        ax1.set_title(f"Total Meltwater Production \n {self.EXP_ID_translation[self.ORIG_ID]} {year}", fontsize=16)
        ax1.set_ylabel('SWE dz')
        ax1.grid()
        handles  = [linehandle_obs, linehandles['Posterior'], patch_handles['Prior']]
        labels = ['SWEobs', 'Posterior', 'Prior']
        ax1.legend(handles, labels)
        plt.show()



        #plot snow melt per grid cell 
        def swe1d_to_melt(swe1d):
            swe1d_melt = swe1d.diff()
            # swe1d_melt = np.where(swe1d_melt > 0, 0, swe1d_melt)*-1
            swe1d_melt = swe1d_melt.mask(swe1d_melt > 0, 0) * -1
            return swe1d_melt
        for ilat in [1, 3, 5]:  # range(len(SWEobs.lat)):
            for ilon in [1, 3, 5]:  # range(len(SWEobs.lon)):
                lat= SWEobs.lat[ilat].values
                lon= SWEobs.lon[ilon].values
                SWEobs_melt_cell = swe1d_to_melt(SWEobs.sel(lat=lat, lon=lon).to_pandas())
                if np.all(np.isnan(SWEobs_melt_cell)):
                    continue                
                elev = int(self.E.dem.sel(lat=lat, lon=lon).values)

                SWE_sims_all_melt_cell = swe1d_to_melt(SWE_sims_all.sel(lat=lat, lon=lon).to_pandas().drop(columns=['spatial_ref', 'lat', 'lon']))
                SWE_sims_post_melt_cell = swe1d_to_melt(SWE_sims_post.sel(lat=lat, lon=lon).to_pandas().drop(columns=['spatial_ref', 'lat', 'lon']))
                SWE_sims_ub_melt_cell = swe1d_to_melt(SWE_sims_UB.sel(lat=lat, lon=lon).to_pandas().drop(columns=['spatial_ref', 'lat', 'lon']))
                
                # Extract temperature data for the grid cell
                temperature_cell = self.meteo_base['tas'].sel(lat=lat, lon=lon).to_pandas()

                # Create the figure and main axis
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
                
                # Plot the SWE melt data
                SWEobs_melt_cell.plot(ax=ax1, color='black', label='SWEobs', linestyle='--', zorder=102)
                SWE_sims_post_melt_cell.plot(ax=ax1, color=colors['Posterior'], alpha=0.2, label='Posterior', linestyle='-')
                SWE_sims_ub_melt_cell.plot(ax=ax1, color=colors['UB'], linestyle='-', alpha = 0.1,
                                           label='Upper benchmark', zorder=1)
                ax1.fill_between(SWE_sims_all_melt_cell.index, SWE_sims_all_melt_cell.min(axis=1), 
                                 SWE_sims_all_melt_cell.max(axis=1), color='grey', alpha=0.2)
                ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
                ax1.set_title(f'Meltwater Production at {ilat}, {ilon} \n Elevation: {elev} m', fontsize=16)
                ax1.set_ylabel('SWE dz')
                ax1.grid()
                ax1.set_ylim(0,100)
                handles = [linehandle_obs, linehandles['Posterior'], linehandles['UB'], patch_handles['Prior']]
                labels = ['SWEobs', 'Posterior', 'Upper benchmark', 'Prior']
                ax1.legend(handles, labels)
                # Create a twin axis for temperature
                ax2 = ax1.twinx()
                temperature_cell.plot(ax=ax2, color='red', label='Temperature', linestyle='-', alpha=0.7)
                ax2.set_ylabel('Temperature (°C)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_xlim(f"{year}-03-01", f"{year}-07-30")

                # Add a legend for temperature
                ax2.legend(loc='upper right')
                ax2.axhline(0, color='red', linestyle='--', alpha=0.5)

                plt.tight_layout()
                plt.show()

        #snowfall plot
        SWEobs_snowfall = swe3d_to_snowfall(SWEobs)
        SWE_sims_all_snowfall = swe3d_to_snowfall(SWE_sims_all)
        SWE_sims_post_snowfall = swe3d_to_snowfall(SWE_sims_post)
        # SWE_sims_UB_snowfall = swe3d_to_snowfall(SWE_sims_UB)
        # SWE_sims_LB_snowfall = swe3d_to_snowfall(SWE_sims_LB)

        f1,ax1 = plt.subplots(1,1,figsize=(10,6))
        handles = [dothandle_obs, linehandle_obs, linehandles['Posterior'], linehandles['UB'], patchhandle_obs]
        labels = ['SWEobs', 'Posterior', 'Upper benchmark', 'Lower benchmark']
        # Plot total melt
        SWEobs_snowfall.plot(ax=ax1, color='black', label='SWEobs', linestyle='--',zorder=102)
        SWE_sims_post_snowfall.plot(ax=ax1, color=colors['Posterior'],
                                    label='_noLegend', legend=None,
                                    alpha = 0.7)
        ax1.fill_between(SWE_sims_all_snowfall.index,
                        SWE_sims_all_snowfall.min(axis=1),
                        SWE_sims_all_snowfall.max(axis=1),
                        color='tab:blue', alpha=0.2)
        # SWE_sims_UB_snowfall.plot(ax=ax1, color=colors['UB'], linestyle='--', label='Upper benchmark',zorder = 1)
        # ax1.fill_between(SWE_sims_all_snowfall.index, SWE_sims_LB_snowfall.min(axis=1), SWE_sims_LB_snowfall.max(axis=1), color=colors['LB'], alpha=0.2)
        ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
        ax1.set_title(f'Total Snowfall Production {year} \n {self.EXP_ID_translation[self.ORIG_ID]}', fontsize=16)
        ax1.set_ylabel('SWE dz')
        ax1.grid()
        ax1.legend([])
        plt.show()

        #2D plots 
        # for SWE_chosen in SWE_targets:
        #     if get_metric_type(SWE_chosen) =='Grid average':
        # self.calc_melt_nse_grid(swe_obs, swe_sim)
        # post_ids = post_ids 
        # for id in post_ids:
        melt_NSE_2d = [self.calc_melt_nse_grid(SWEobs, self.SWE[YEAR][id]) for id in post_ids]
        melt_NSE_2d_joint = xr.concat(melt_NSE_2d, dim = 'id')
        melt_NSE_2d_median = melt_NSE_2d_joint.median(dim = 'id')

        ub_ids = SWE_PRI_dic['SWE_melt_NSE_grid']['Q_UB'][YEAR].columns
        melt_NSE_2d_UB = [self.calc_melt_nse_grid(SWEobs, self.SWE[YEAR][id]) for id in ub_ids]
        melt_NSE_2d_UB_joint = xr.concat(melt_NSE_2d_UB, dim = 'id')
        melt_NSE_2d_UB_median = melt_NSE_2d_UB_joint.median(dim = 'id')
        
        f1,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6),sharey = True)
        melt_NSE_2d_median.plot(ax=ax1,vmin = -1, vmax = 1,cmap = 'RdBu')
        ax1.set_title(f'Melt NSE Posterior {year} \n {self.EXP_ID_translation[self.ORIG_ID]}')
        melt_NSE_2d_UB_median.plot(ax=ax2,vmin = -1, vmax = 1,cmap = 'RdBu')
        ax2.set_title(f'Melt NSE 2D Upper Benchmark {year} \n {self.EXP_ID_translation[self.ORIG_ID]} \n SWE target = {SWE_chosen}')

        timeslice = slice(f'{year}-04-01',f'{year}-09-30')
        QQall = self.Q.loc[timeslice]
        QQmin = QQall.min(axis = 1)
        QQmax = QQall.max(axis = 1)

        QQpost = self.Q.loc[timeslice][post_ids]
        QQobs = self.Qobs.loc[timeslice]
        # QQ_UB = SWE_PRI_dic[SWE_chosen]['Q_UB'][year][timeslice]
        # QQ_UB = self.Q.loc[timeslice][UB_id]

        f1,ax1 = plt.subplots(figsize = (7,4))
        try:
            QQpost.plot(ax = ax1, color = colors['Posterior'],
                        alpha = alphas['Posterior'],legend = None,
                        zorder = 50,
                        linewidth = 0.5)
        except:
            print(f"Year {year} has no posterior runs")
            pass
        QQobs.plot(ax = ax1,color= 'black', linestyle = '--', legend = None,zorder = 100)
        # QQ_UB.plot(ax = ax1, color = colors['UB'],alpha = alphas['UB'], linestyle = '-', legend = None)
        ax1.fill_between(QQall.index, QQmin, QQmax, color = colors['Prior'], alpha = 0.2)
        handles = [linehandle_obs,linehandles['Posterior'], linehandles['UB'], patch_handles['Prior']]
        labels = ['Observed', 'Posterior', 'Upper benchmark', 'Prior']
        # labels = ['Posterior','Obs', 'Upper benchmark', 'Prior']
        ax1.legend(handles, labels)
        ax1.grid()
        ax1.set_title(f"Q for {year} \n {self.EXP_ID_translation[self.ORIG_ID]}", fontsize=16)
        ax1.set_ylabel('Q [m3/s]')
        ax1.set_ylim(bottom = 0, top = ax1.get_ylim()[1]/2)
        ax1.set_xlabel(None)

#%%
# Get all parameters across years
# for SWE_chosen in SWE_targets:
for self in LOA_objects.values():
    if 'LB' in self.ORIG_ID:
        continue
        
    SWE_chosen = 'SWE_melt_NSE_grid'
    all_params = list(self.pars[self.START_YEAR].columns)
    SEL = self.SWE_PRI_dic[SWE_chosen]['SWE_results'][self.SWE_PRI_dic[SWE_chosen]['SWE_results']['Q_metric'] == Q_chosen]

    # Loop through each parameter
    melted_params = pd.DataFrame(columns = ['id','param','param_kind','category','value','year'])
    for i, param in enumerate(all_params):
        param_kind = self.param_kinds[param]
        
        # Gather parameter values across years
        for year in range(self.START_YEAR, self.END_YEAR + 1):
            # Get values for this parameter in this year
            values = self.pars[year][param]
            
            # Normalize values to [0, 1] based on min and max across all years
            param_min = self.param_ranges[param][0]
            param_max = self.param_ranges[param][1]
            if param_max > param_min:  # Avoid division by zero
                norm_values = (values - param_min) / (param_max - param_min)
            else:
                norm_values = np.zeros_like(values)

            # prior = pd.DataFrame({'id': norm_values.index, 
            #                       'param': param,
            #                                       'category': 'Prior',
            #                                         'value': norm_values.values,
            #                                           'year': year})
                
            posterior_ids = SEL[SEL['year'] == year][SEL['category'] == 'Posterior']['index']
            posterior_vals = norm_values[posterior_ids]
            posterior = pd.DataFrame({'id': posterior_ids,
                                    'param': param,
                                    'param_kind': param_kind,
                                                    'category': 'Posterior',
                                                    'value': posterior_vals.values,
                                                    'year': year})  
            melted_params = pd.concat([melted_params, posterior], ignore_index=True)

            ub_ids = SEL[SEL['year'] == year][SEL['category'] == 'UB']['index']
            ub_vals = norm_values[ub_ids]
            ub = pd.DataFrame({'id': ub_ids,
                                        'param': param,
                                        'param_kind': param_kind,
                                                        'category': 'UB',
                                                        'value': ub_vals.values,
                                                        'year': year})
            melted_params = pd.concat([melted_params, ub], ignore_index=True)
    self.melted_params = melted_params


#%%
# def all_years_param_plot(data,category = 'Posterior'
#                         ):
import matplotlib.colors as mcolors

years = list(range(2001, 2023))
self = LOA_objects['Syn_244a']
sfcf_values = [self.trueparams[f'sfcf_{year}'] for year in years]
category = 'Posterior'
# Normalize the sfcf values to the range [0, 1]
norm = mcolors.Normalize(vmin=min(sfcf_values), vmax=max(sfcf_values))

# Choose a qualitative colormap
cmap = plt.cm.coolwarm_r  # You can also try 'Set3', 'Paired', etc.

# Map the normalized values to colors
colors_params = [cmap(norm(value)) for value in sfcf_values]
param_trans = dict(meteo = r'$\theta_{meteo}$',
                    snow = r'$\theta_{snow}$',
                sfcf = r'SFCF',
                    sfcf_scale = r'SFCF_ELEV',
                    rfcf = r'RFCF',
                    TT = r'TT',
                    tt_scale = r'TT_ELEV',
                    m_hock = r'M_HOCK',
                    r_hock = r'R_HOCK',
                    mwf = r'MWF',
                    WHC = r'WHC',
                    CV = r'CV',)

# if not param_kind =='all':
#     data = data[data['param_kind']==param_kind]
# data = data[data['category'] == category]

# f1,ax1 = plt.subplots(1,1,figsize = (2,10))
f1,axes= plt.subplots(1,4,figsize = (8,8),sharey=False,sharex = True)
plt.subplots_adjust(wspace=0.25)
# for selfi,self in enumerate(LOA_objects.values()):
for selfi, self in enumerate(LOA_objects.values()):
    print(self.ORIG_ID)
    if 'LB' in self.ORIG_ID:
        continue
    data = self.melted_params
    for axi,param_kind in enumerate(['meteo','snow']):
        ax = axes[axi+ selfi*2] 
        data_ = data[data['param_kind'] == param_kind]
        bxplt = sns.boxenplot(ax = ax,data = data_, y = 'param',orient = 'h',
                    x = 'value', hue = 'year', dodge = True,
                    palette =colors_params,linewidth = 0.4,
                    line_kws=dict(linewidth=3, color="white"),
                    flier_kws = dict(alpha = 0.5,s = 0.3),
                    alpha = 0.9, legend=(param_kind =='snow'),
                    width = 0.9,gap =0.0,
                    k_depth=1)
        # bxplt = sns.boxplot(ax = ax,data = data_, y = 'param',orientation = 'horizontal',
        #                     x = 'value', hue = 'year', dodge = True,
        #                     palette =colors,legend=(param_kind =='snow'), 
        #                     fliersize = 0.3, saturation = 0.9, 
        #                     width = 0.9,gap =0.0, linewidth = 0.5)

        unique_params =data_['param'].unique()
        ymin_axis, ymax_axis = ax.get_ylim()
        y_range = ymax_axis - ymin_axis

        # For each parameter, add a vertical line at the true value
        # if self.ORIG_ID =='Syn_34b':
        for i, param in enumerate(unique_params):
            if param != 'sfcf':  # Skip sfcf parameter
                if (self.ORIG_ID =='Syn_244b') & (not param =='rfcf'):
                    continue
                # Get the true value and normalize it
                truevalue = self.trueparams[param]
                norm_truevalue = (truevalue - self.param_ranges[param][0]) / (self.param_ranges[param][1] - self.param_ranges[param][0])
                # Get the y-position of this parameter (i is the position in the unique_params list)
                # param_y_pos = unique_params.shape[0] - 1 - i  # Reverse order to match boxenplot
                param_y_pos = i
                # if not param_kind =='snow':
                #     param_y_pos +=1
                # Calculate normalized positions for the line segment
                # Convert axis position to relative position (0 to 1)
                y_rel_pos = (param_y_pos - ymin_axis) / y_range
                y_height = 0.9 / len(unique_params)  # Height of line segment as fraction of y-axis
                
                # Draw the vertical line at the true parameter value
                # ax.axvline(norm_truevalue, ymin=y_rel_pos - y_height/2, ymax=y_rel_pos + y_height/2,
                #             color='black', linestyle='--', linewidth=2)
                
                par_df = pd.DataFrame(index = years,
                                    columns = ['value'])
                for year in years:
                    par_df.loc[year] = norm_truevalue
                par_df['param'] = param
                par_df['year'] = years
                sns.pointplot(ax = ax,data=par_df,x='value',hue = 'year',
                            y = 'param',
                            marker = 'x', palette = 'dark:black',legend = False,
                            markersize = 5,markeredgewidth = 1,dodge = 0.87,)
            
            else:
                if self.ORIG_ID == 'Syn_244b':
                    continue
                sfcf_true = [par for par in self.trueparams if 'sfcf_2' in par]
                sfcf_dic = {par: self.trueparams[par] for par in sfcf_true}
                sfcf_years = [int(param.split('_')[1]) for param in sfcf_true]
                sfcf_df = pd.DataFrame(index = sfcf_years, 
                                        columns = ['value'])
                for year in sfcf_years:
                    sfcf_df.loc[year] = (sfcf_dic[f'sfcf_{year}'] - self.param_ranges['sfcf'][0]) / (self.param_ranges['sfcf'][1] - self.param_ranges['sfcf'][0])
                sfcf_df['param'] = 'sfcf'
                sfcf_df['year'] = sfcf_years
                sns.pointplot(ax = ax,data=sfcf_df,
                            x = 'value', 
                            y = 'param', 
                            hue = 'year',
                            marker = 'x',palette = 'dark:black',
                            markersize = 5,markeredgewidth = 1,dodge = 0.87,
                            legend = False)
        y_ticklabels = ax.get_yticklabels()

        # Translate the labels using the param_trans dictionary
        translated_labels = [param_trans[label.get_text()] if label.get_text() in param_trans else label.get_text() for label in y_ticklabels]
        
        # Set the translated labels back to the axis
        ax.set_yticklabels(translated_labels,rotation = 90, va = 'center')
        ax.grid(alpha = 0.5, axis = 'x',color = 'black')
        ax.set_title(f'{param_trans[param_kind]}', fontsize=10)

        for spine in ax.spines.values():
            spine.set_color('tab:blue' if self.EXP_ID_translation[self.ORIG_ID] =='Fully Synthetic' else 'tab:orange')
            spine.set_linewidth(1.5)
            spine.set_alpha(0.8)


        if ax != axes[-1]:
            ax.set_ylabel('')    
            ax.set_xlabel('')    
            # ax.set_xlabel('                                            Normalized parameter value')
            ax.legend([''],edgecolor = 'white', loc = (1.01,0.2), ncol = 1)
        else:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.legend(loc = (1.1,0.2), ncol = 1)


            legend_handle = plt.Line2D([0], [0], marker='x', color='k',linestyle='None',)
            legend_label = r'$\theta^*$'
            current_legend = ax.get_legend()

    # Get current handles and labels from the legend
            if current_legend:
                handles = current_legend.legend_handles
                labels = [text.get_text() for text in current_legend.get_texts()]
            else:
                handles = []
                labels = []

            # Append the new handle and label
            handles.append(legend_handle)
            labels.append(legend_label)

            # Create a new legend with the combined handles and labels
            ax.legend(handles, labels,loc = (1.1,0.1), ncol = 1)
            ax.set_xlim(0,1)
            ax.tick_params(axis='y', pad = 0)

    fontcol = 'tab:blue' if self.ORIG_ID == 'Syn_244a' else 'tab:orange'
    text = f"<{self.EXP_ID_translation[self.ORIG_ID]}>"# \n {category} parameter ranges"
    from highlight_text import fig_text
    fig_text(
        x=0.31+selfi*0.41 , # position on x-axis (centered)
        y=0.935, # position on y-axis
        s=text,
        fontsize=12,
        ha='center',
        textalign = 'center',
        highlight_textprops=[{'color': fontcol, 'fontweight': 'bold'}]
    )
    subfigs = ['(a)','(b)','(c)','(d)']
    x_positions = [0.135+selfi*0.205 for selfi in range(4)]
    for ss in range(len(subfigs)):
        fig_text(
            x=x_positions[ss],
            y=0.905,
            s=subfigs[ss],
            fontsize=10,
            ha='center',
        )



# f1.suptitle(f'{category} parameter ranges', 
#             fontsize=14, y=0.975)
fig_text(x = 0.55,y = 0.06,s = 'Normalized parameter value [-]',
         fontsize = 10, ha = 'center', va = 'center')
# f1.savefig(join(self.FIGDIR, f'FF_FS_{category}_bothparamkinds_parameters.png'),
#             bbox_inches='tight', dpi=300)
f1.savefig(join("/home/pwiersma/scratch/Figures/Paper1", f'Fig2.svg'),
            bbox_inches='tight')


#%%
# swecorr_dic = {}
# for self in LOA_objects.values():
#     if 'LB' in self.ORIG_ID:
#         continue
#     swemetrics = pd.concat([self.swemetrics[year] for year in range(self.START_YEAR, self.END_YEAR+1)],axis=0)
#     swemetrics = swemetrics[self.swe_metrics_to_use]
#     Qmetrics = pd.concat([self.Qmetrics[year] for year in range(self.START_YEAR, self.END_YEAR+1)],axis=0)
#     swemetrics['Q_NSE'] = Qmetrics['NSE_meltseason']

#     swemetrics = swemetrics.rename(columns = 
#             {col: translate_SWE_metric_name(col,keep_suffix=True) for col in swemetrics.columns})

#     for col in swemetrics.columns: 
#         print(col)
    
#         if 'ELEV' in col:
#             #remove col from swemetrics
#             swemetrics = swemetrics.drop(col, axis = 1)
#         if 'Q' in col:
#             #remove col from swemetrics
#             swemetrics = swemetrics.rename(columns = {'AGG-Q ':'Q'})
#         # else:

#     new_order = ['Q',
#                 'AGG-Melt',
#                 'AGG-Snowfall',
#                 'AGG-Accumulation ',
#                 'AGG-Melt-out ',
#                 'AGG-Onset ',
#                 'GRID-Melt',
#                 'GRID-Snowfall',
#                 'GRID-Accumulation ',
#                 'GRID-Melt-out ',
#                 'GRID-Onset ']
#     swemetrics = swemetrics[new_order]

#     swecorr = swemetrics.corr('spearman').abs()
#     swecorr_dic[self.ORIG_ID] = swecorr
# # sns.heatmap(swecorr, cmap = 'coolwarm',
# #             cbar_kws={'label': 'Absolute pearson correlation'},
# #             vmin = -1, vmax = 1,)
# # plt.title('Correlation between SWE metrics over all years \n (Synthetic case study 2001-2022 in Dischma)')


swecorr_dic = {}
for self in LOA_objects.values():
    if 'LB' in self.ORIG_ID:
        continue
    
    yearly_corrs = []  # List to store correlation matrices for each year
    
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        # Extract SWE metrics and Q metrics for the current year
        swemetrics = self.swemetrics[year][self.swe_metrics_to_use]
        Qmetrics = self.Qmetrics[year]
        
        # Add Q_NSE column to SWE metrics
        # swemetrics['Q'] = Qmetrics['NSE_meltseason']
        # swemetrics['QBias'] = Qmetrics['Qmean_meltseason_ME']
        # swemetrics['QKGE'] = Qmetrics['KGE_meltseason']
        
        # Rename columns using the translate_SWE_metric_name function
        swemetrics = swemetrics.rename(columns={
            col: translate_SWE_metric_name(col, keep_suffix=True) for col in swemetrics.columns
        })
        swemetrics['Q-NSE'] = Qmetrics['NSE_meltseason']

        
        # Remove columns containing 'ELEV' and rename 'AGG-Q' to 'Q'
        for col in swemetrics.columns:
            if 'ELEV' in col:
                swemetrics = swemetrics.drop(col, axis=1)
            # if 'Q' in col:
            #     swemetrics = swemetrics.rename(columns={'AGG-Q ': 'Q-NSE'})
                # swemetrics = swemetrics.rename(columns={'AGG-QNSE': 'Q-NSE',
                #                                         'AGG-QBias': 'Q-Bias',
                #                                         'AGG-QKGE': 'Q-KGE'})
        
        # Reorder columns
        new_order = [
            'Q-NSE',
            # 'Q-Bias',
            # 'Q-KGE',
            'AGG-Melt',
            'AGG-Snowfall',
            'AGG-Accumulation ',
            'AGG-Melt-out ',
            'AGG-Onset ',
            'GRID-Melt',
            'GRID-Snowfall',
            'GRID-Accumulation ',
            'GRID-Melt-out ',
            'GRID-Onset '
        ]
        swemetrics = swemetrics[new_order]
        
        # Calculate Spearman correlation matrix for the current year
        yearly_corr = swemetrics.corr('spearman').abs()
        yearly_corrs.append(yearly_corr)
    
    # Take the median correlation matrix across all years without losing the order
    swecorr = pd.concat(yearly_corrs).groupby(level=0).median()
    #make sure columns and headers are in the same order
    swecorr = swecorr.reindex(columns=new_order, index=new_order)
    # swecorr = sum(yearly_corrs) / len(yearly_corrs)
    swecorr_dic[self.ORIG_ID] = swecorr

mask1 = np.tril(np.ones_like(swecorr, dtype=bool))
mask2 = np.triu(np.ones_like(swecorr, dtype=bool))


# plt.figure()
# # swecorr1 = swemetrics.corr('spearman').abs()
# sns.heatmap(swecorr_dic['Syn_244a'],cmap = 'Blues',
#             cbar_kws={'label': 'Spearman rank correlation [-]'}, 
#             vmin = 0, vmax = 1, mask = mask1,annot = True)   
# sns.heatmap(swecorr_dic['Syn_244b'],cmap = 'Oranges' ,
#             cbar_kws={'label': 'Spearman rank correlation [-]'}, 
#             vmin = 0, vmax = 1, mask = mask2,annot = True)   
# plt.title(f'Correlation between all Q and SWE metrics over all years')


#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Assume swecorr_dic is already populated, and keys 'Syn_244a', 'Syn_244b' exist
keys = ['Syn_244a', 'Syn_244b']
matA = swecorr_dic[keys[0]]
matB = swecorr_dic[keys[1]]
labels = matA.columns.tolist()

# Strip prefixes for display
short_labels = ['NSE'] + [lab.split('-', 1)[1].strip() for lab in labels[1:]]

# Build masks
maskA = np.tril(np.ones_like(matA, dtype=bool))
maskB = np.triu(np.ones_like(matB, dtype=bool))

fig, ax = plt.subplots(figsize=(9,7))

# Create ScalarMappable objects for colorbars
smA = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='Blues')
smB = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='Oranges')

# Lower triangle in Blues, no colorbar
sns.heatmap(matA, mask=maskA, annot=True,
            cmap='Blues', vmin=0, vmax=1,
            cbar=False, ax=ax)

# Upper triangle in Oranges, no colorbar
sns.heatmap(matB, mask=maskB, annot=True,
            cmap='Oranges', vmin=0, vmax=1,
            cbar=False, ax=ax)

# Strip prefixes from tick labels
ax.set_xticks(np.arange(len(short_labels)) + 0.5)
ax.set_xticklabels(short_labels, rotation=45, ha='right')
ax.set_yticks(np.arange(len(short_labels)) + 0.5)
ax.set_yticklabels(short_labels, rotation=0)

ax.axhline(y=1, color='k', lw=2, alpha=0.5, zorder=10)  # horizontal line below first row
ax.axvline(x=1, color='k', lw=2, alpha=0.5, zorder=10)  # vertical line after first column

ax.axhline(y=6, color='k', lw=2, alpha=0.5, zorder=10)  # horizontal line below first row
ax.axvline(x=6, color='k', lw=2, alpha=0.5, zorder=10)  # vertical line after first column

from matplotlib.patches import Rectangle

# Outline diagonal from (1,6) to (5,10)
for i in range(1, 6):  # rows 1 to 5
    j = i + 5          # cols 6 to 10
    rect = Rectangle((j, i), 1, 1, fill=False,
     edgecolor='k', lw=2, zorder=20, alpha = 0.5)
    ax.add_patch(rect)

# Outline diagonal from (6,1) to (10,5)
for i in range(6, 11):  # rows 6 to 10
    j = i - 5           # cols 1 to 5
    rect = Rectangle((j, i), 1, 1, fill=False, 
    edgecolor='k', lw=2, zorder=20, alpha = 0.5)
    ax.add_patch(rect)

N = len(short_labels)
# ax.set_xlim(-1.5, N)        # make room on the left
# ax.invert_yaxis()           # invert so row 0 is at top

# groups = {'AGG': (1, 5), 'GRID': (6, 10)}
# groups = {r'Q':(0,0),r'AGG': (1, 5), 
# r'GRID': (6, 10)}
groups = {r'$E_{Q}$':(0,0),r'$E_{AGG}$': (1, 5), 
r'$E_{GRID}$': (6, 10)}

# Row brackets (on left)
for label, (start, end) in groups.items():
    ax.annotate(
        '', 
        xy=(-0.5, start + 0.5), 
        xytext=(-0.5, end + 0.5),
        xycoords='data', textcoords='data',
        arrowprops=dict(arrowstyle='-[,widthB=6.0', lw=2),
        zorder=500
    )
    ax.text(
        -3.0,
        (start + 0.5 + end + 0.5) / 2,
        label,
        va='center', ha='center',
        fontsize=13,
        zorder=5
    )

# Column brackets (on top)
for label, (start, end) in groups.items():
    ax.annotate(
        '', 
        xy=(start + 0.5, -0.5), 
        xytext=(end + 0.5, -0.5),
        xycoords='data', textcoords='data',
        arrowprops=dict(arrowstyle='-[,widthB=6.0', lw=2),
        zorder=5
    )
    ax.text(
        (start + 0.5 + end + 0.5) / 2,
        13,
        label,
        ha='center', va='top',
        fontsize=13,
        zorder=5
    )

# Two colorbars, only the second labeled
caxA = fig.add_axes([0.8, 0.35, 0.02, 0.4])
caxB = fig.add_axes([0.82, 0.35, 0.02, 0.4])

cbA = fig.colorbar(smA, cax=caxA, ticks=[0, 1])
cbA.ax.set_ylabel('')
cbA.ax.set_yticklabels([])

cbB = fig.colorbar(smB, cax=caxB, ticks=[0, 1], label='Spearman rank correlation [-]')

# ax.set_title('Median annual performance metric correlations')
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make room for colorbars
# plt.savefig(join(self.FIGDIR, f'FF_FS_Correlation_between_all_Q_and_SWE_metrics_over_all_years.png'),
#             bbox_inches='tight', dpi=300)
#save in eps
plt.savefig(join("/home/pwiersma/scratch/Figures/Paper1", f'Fig5_E.svg'),
            bbox_inches='tight')
plt.show()

#%%

# assume swecorr_dic already populated, and keys 'Syn_244a','Syn_244b' exist
# keys = ['Syn_244a','Syn_244b']
# matA = swecorr_dic[keys[0]]
# matB = swecorr_dic[keys[1]]
# labels = matA.columns.tolist()

# # strip prefixes for display
# short_labels =['Q-NSE'] + [lab.split('-',1)[1].strip() for lab in labels[1:]]

# # build masks
# maskA = np.tril(np.ones_like(matA, dtype=bool))
# maskB = np.triu(np.ones_like(matB, dtype=bool))

# fig, ax = plt.subplots(figsize=(9,7))

# # lower‐triangle in Blues, no colorbar
# sns.heatmap(matA, mask=maskA,annot=True,
#             cmap='Blues', vmin=0, vmax=1,
#             cbar=False, ax=ax)

# # upper‐triangle in Oranges, no colorbar
# sns.heatmap(matB, mask=maskB,annot = True,
#             cmap='Oranges', vmin=0, vmax=1,
#             cbar=False, ax=ax)

# # strip prefixes from tick labels
# ax.set_xticks(np.arange(len(short_labels)) + 0.5)
# ax.set_xticklabels(short_labels, rotation=45, ha='right')
# ax.set_yticks(np.arange(len(short_labels)) + 0.5)
# ax.set_yticklabels(short_labels, rotation=0)

# N = len(short_labels)
# # ax.set_xlim(-1.5, N)        # make room on the left
# # ax.invert_yaxis()           # invert so row 0 is at top

# groups = {'AGG': (1, 5), 'GRID': (6, 10)}

# # Row brackets (on left)
# for label, (start, end) in groups.items():
#     ax.annotate(
#         '', 
#         xy=(-0.5, start+0.5), 
#         xytext=(-0.5, end+0.5),
#         xycoords='data', textcoords='data',
#         arrowprops=dict(arrowstyle='-[,widthB=6.0', lw=2),
#         zorder=500
#     )
#     ax.text(
#         -3.0,
#         (start+0.5 + end+0.5)/2,
#         label,
#         va='center', ha='center',
#         fontsize=12,
#         zorder=5
#     )

# # Column brackets (on top)
# for label, (start, end) in groups.items():
#     ax.annotate(
#         '', 
#         xy=(start+0.5, -0.5), 
#         xytext=(end+0.5, -0.5),
#         xycoords='data', textcoords='data',
#         arrowprops=dict(arrowstyle='-[,widthB=6.0', lw=2),
#         zorder=5
#     )
#     ax.text(
#         (start+0.5 + end+0.5)/2,
#         13,
#         label,
#         ha='center', va='top',
#         fontsize=12,
#         zorder=5
#     )

# # two colorbars, only the second labeled
# caxA = fig.add_axes([0.8, 0.3, 0.02, 0.4])
# caxB = fig.add_axes([0.82, 0.3, 0.02, 0.4])

# cbA = fig.colorbar(smA, cax=caxA, ticks=[0,1])
# cbA.ax.set_ylabel('')
# cbA.ax.set_yticklabels([])
# # no need to set ticks position explicitly—by default they’re on the left of that tiny box

# cbB = fig.colorbar(smB, cax=caxB, ticks=[0,1], label='Spearman rank correlation [-]')


# # ax.set_title('2001-2022 median annual correlations over prior ensemble')
# plt.tight_layout(rect=[0, 0, 0.8, 1])  # make room for colorbars
# plt.show()

#%%
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import seaborn as sns
# import matplotlib as mpl

# # Assume you already have:
# # - swecorr_dic['Syn_244a'] and ['Syn_244b']
# # - labels with 'AGG-' or 'GRID-' prefixes
# # - norm, smA, smB

# # Get matrix and labels
# matA = swecorr_dic['Syn_244a']
# matB = swecorr_dic['Syn_244b']
# labels = matA.columns.tolist()
# short_labels =['Q']+ [label.split('-', 1)[-1].strip() for label in labels[1:]]

# fig, ax = plt.subplots(figsize=(10, 9))
# sns.heatmap(matA, mask=np.tril(np.ones_like(matA, bool)), cmap='Blues', vmin=0, vmax=1, cbar=False, ax=ax)
# sns.heatmap(matB, mask=np.triu(np.ones_like(matB, bool)), cmap='Oranges', vmin=0, vmax=1, cbar=False, ax=ax)

# # Format axis labels
# ax.set_xticks(np.arange(len(short_labels)) + 0.5)
# ax.set_xticklabels(short_labels, rotation=45, ha='right')
# ax.set_yticks(np.arange(len(short_labels)) + 0.5)
# ax.set_yticklabels(short_labels, rotation=0)

# # Bracket drawing function using patches
# # def draw_bracket(ax, start, end, axis='y', label='', offset=0.8):
# #     """
# #     Draw a bracket along x or y axis and add text label beside it.
# #     """
# #     width = end - start + 1
# #     if axis == 'y':
# #         # Vertical bracket on left

# #         rect = patches.FancyArrowPatch(
# #             posA = 0.5, posB = 0.4, 
# #             arrowstyle="]-[", 
# #         )
# #         ax.add_patch(rect)
# #         # rect = patches.FancyBboxPatch(
# #         #     (-1.8, start), 0.4, width,
# #         #     boxstyle="square", ec="black", fc="none", lw=2
# #         # )
# #         # ax.add_patch(rect)
# #         # ax.text(-2.1, start + width / 2, label, va='center', ha='center', fontsize=12, rotation=90)
# #     else:
# #         # Horizontal bracket on top
# #         rect = patches.FancyBboxPatch(
# #             (start, -1.8), width, 0.4,
# #             boxstyle="square", ec="black", fc="none", lw=2
# #         )
# #         ax.add_patch(rect)
# #         ax.text(start + width / 2, -2.1, label, va='top', ha='center', fontsize=12)

# # # Apply brackets
# # agg_indices  = (0, 4)
# # grid_indices = (5, 9)
# # draw_bracket(ax, *agg_indices, axis='y', label='AGG')
# # draw_bracket(ax, *grid_indices, axis='y', label='GRID')
# # draw_bracket(ax, *agg_indices, axis='x', label='AGG')
# # draw_bracket(ax, *grid_indices, axis='x', label='GRID')

# # Axis limits to fit brackets
# ax.set_xlim(-2.2, len(labels))
# ax.set_ylim(len(labels), -2.2)

# # Colorbars (side-by-side)
# cax1 = fig.add_axes([0.85, 0.15, 0.015, 0.7])
# cax2 = fig.add_axes([0.87, 0.15, 0.015, 0.7])
# cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Blues'), cax=cax1, ticks=[0, 1])
# cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Oranges'), cax=cax2, ticks=[0, 1])
# cb2.set_label('Spearman rank correlation [-]')
# cb1.ax.set_ylabel('')
# cb1.ax.set_yticklabels([])
# cb2.ax.yaxis.set_ticks_position('right')
# cb1.ax.yaxis.set_ticks_position('right')

# ax.set_title("Correlation between all Q and SWE metrics over all years")
# plt.tight_layout(rect=[0, 0, 0.82, 1])
# plt.show()


#%%

    # plt.close(f1)
# all_years_param_plot(melted_params,category = 'Posterior')


# all_years_param_plot(melted_params,category = 'Posterior',param_kind ='meteo')
# all_years_param_plot(melted_params,category = 'Posterior',param_kind ='snow')
# all_years_param_plot(melted_params,category = 'UB',param_kind ='meteo')
# all_years_param_plot(melted_params,category = 'UB',param_kind ='snow')
# def all_years_param_plot(data,category = 'Posterior'
#                             ):
#         import matplotlib.colors as mcolors

#         years = list(range(2001, 2023))
#         sfcf_values = [self.trueparams[f'sfcf_{year}'] for year in years]

#         # Normalize the sfcf values to the range [0, 1]
#         norm = mcolors.Normalize(vmin=min(sfcf_values), vmax=max(sfcf_values))

#         # Choose a qualitative colormap
#         cmap = plt.cm.coolwarm_r  # You can also try 'Set3', 'Paired', etc.

#         # Map the normalized values to colors
#         colors = [cmap(norm(value)) for value in sfcf_values]
#         param_trans = dict(meteo = r'$\theta_{M}$',
#                             snow = r'$\theta_{snow}$',
#                         sfcf = r'SFCF',
#                             sfcf_scale = r'SFCF_ELEV',
#                             rfcf = r'RFCF',
#                             TT = r'TT',
#                             tt_scale = r'TT_ELEV',
#                             m_hock = r'M_HOCK',
#                             r_hock = r'R_HOCK',
#                             mwf = r'MWF',
#                             WHC = r'WHC',
#                             CV = r'CV',)

#         # if not param_kind =='all':
#         #     data = data[data['param_kind']==param_kind]
#         # data = data[data['category'] == category]
        
#         # f1,ax1 = plt.subplots(1,1,figsize = (2,10))
#         f1,axes= plt.subplots(1,2,figsize = (4,8),sharey=False,sharex = True)
#         # plt.subplots_adjust(wspace=0.55)
#         for axi,param_kind in enumerate(['meteo','snow']):
#             ax = axes[axi]
#             data_ = data[data['param_kind'] == param_kind]
#             bxplt = sns.boxenplot(ax = ax,data = data_, y = 'param',orient = 'h',
#                         x = 'value', hue = 'year', dodge = True,
#                         palette =colors,linewidth = 0.4,
#                         line_kws=dict(linewidth=3, color="white"),
#                         flier_kws = dict(alpha = 0.5,s = 0.3),
#                         alpha = 0.9, legend=(param_kind =='snow'),
#                         width = 0.9,gap =0.0,
#                         k_depth=1)
#             # bxplt = sns.boxplot(ax = ax,data = data_, y = 'param',orientation = 'horizontal',
#             #                     x = 'value', hue = 'year', dodge = True,
#             #                     palette =colors,legend=(param_kind =='snow'), 
#             #                     fliersize = 0.3, saturation = 0.9, 
#             #                     width = 0.9,gap =0.0, linewidth = 0.5)

#             unique_params =data_['param'].unique()
#             ymin_axis, ymax_axis = ax.get_ylim()
#             y_range = ymax_axis - ymin_axis

#             # For each parameter, add a vertical line at the true value
#             # if self.ORIG_ID =='Syn_34b':
#             for i, param in enumerate(unique_params):
#                 if param != 'sfcf':  # Skip sfcf parameter
#                     if (self.ORIG_ID =='Syn_244b') & (not param =='rfcf'):
#                         continue
#                     # Get the true value and normalize it
#                     truevalue = self.trueparams[param]
#                     norm_truevalue = (truevalue - self.param_ranges[param][0]) / (self.param_ranges[param][1] - self.param_ranges[param][0])
#                     # Get the y-position of this parameter (i is the position in the unique_params list)
#                     # param_y_pos = unique_params.shape[0] - 1 - i  # Reverse order to match boxenplot
#                     param_y_pos = i
#                     # if not param_kind =='snow':
#                     #     param_y_pos +=1
#                     # Calculate normalized positions for the line segment
#                     # Convert axis position to relative position (0 to 1)
#                     y_rel_pos = (param_y_pos - ymin_axis) / y_range
#                     y_height = 0.9 / len(unique_params)  # Height of line segment as fraction of y-axis
                    
#                     # Draw the vertical line at the true parameter value
#                     # ax.axvline(norm_truevalue, ymin=y_rel_pos - y_height/2, ymax=y_rel_pos + y_height/2,
#                     #             color='black', linestyle='--', linewidth=2)
                    
#                     par_df = pd.DataFrame(index = years,
#                                         columns = ['value'])
#                     for year in years:
#                         par_df.loc[year] = norm_truevalue
#                     par_df['param'] = param
#                     par_df['year'] = years
#                     sns.pointplot(ax = ax,data=par_df,x='value',hue = 'year',
#                                 y = 'param',
#                                 marker = 'x', palette = 'dark:black',legend = False,
#                                 markersize = 5,markeredgewidth = 1,dodge = 0.87,)
                
#                 else:
#                     if self.ORIG_ID == 'Syn_244b':
#                         continue
#                     sfcf_true = [par for par in self.trueparams if 'sfcf_2' in par]
#                     sfcf_dic = {par: self.trueparams[par] for par in sfcf_true}
#                     sfcf_years = [int(param.split('_')[1]) for param in sfcf_true]
#                     sfcf_df = pd.DataFrame(index = sfcf_years, 
#                                             columns = ['value'])
#                     for year in sfcf_years:
#                         sfcf_df.loc[year] = (sfcf_dic[f'sfcf_{year}'] - self.param_ranges['sfcf'][0]) / (self.param_ranges['sfcf'][1] - self.param_ranges['sfcf'][0])
#                     sfcf_df['param'] = 'sfcf'
#                     sfcf_df['year'] = sfcf_years
#                     sns.pointplot(ax = ax,data=sfcf_df,
#                                 x = 'value', 
#                                 y = 'param', 
#                                 hue = 'year',
#                                 marker = 'x',palette = 'dark:black',
#                                 markersize = 5,markeredgewidth = 1,dodge = 0.87,
#                                 legend = False)
#             y_ticklabels = ax.get_yticklabels()

#             # Translate the labels using the param_trans dictionary
#             translated_labels = [param_trans[label.get_text()] if label.get_text() in param_trans else label.get_text() for label in y_ticklabels]
            
#             # Set the translated labels back to the axis
#             ax.set_yticklabels(translated_labels,rotation = 90, va = 'center')
#             ax.grid(alpha = 0.5, axis = 'x',color = 'black')
#             if axi ==0:
#                 ax.set_ylabel('')        
#                 ax.set_xlabel('                                            Normalized parameter value')
#                 ax.legend([''],edgecolor = 'white', loc = (1.01,0.2), ncol = 1)
#             else:
#                 ax.set_ylabel('')
#                 ax.set_xlabel('')
#                 ax.legend(loc = (1.1,0.2), ncol = 1)


#                 legend_handle = plt.Line2D([0], [0], marker='x', color='k',linestyle='None',)
#                 legend_label = r'$\theta^*$'
#                 current_legend = ax.get_legend()

#     # Get current handles and labels from the legend
#                 if current_legend:
#                     handles = current_legend.legend_handles
#                     labels = [text.get_text() for text in current_legend.get_texts()]
#                 else:
#                     handles = []
#                     labels = []

#                 # Append the new handle and label
#                 handles.append(legend_handle)
#                 labels.append(legend_label)

#                 # Create a new legend with the combined handles and labels
#                 ax.legend(handles, labels,loc = (1.1,0.1), ncol = 1)
#             ax.set_title(f'{param_trans[param_kind]}', fontsize=10)
#             ax.set_xlim(0,1)
#             ax.tick_params(axis='y', pad = 0)

#         fontcol = 'tab:blue' if self.ORIG_ID == 'Syn_244a' else 'tab:orange'
#         text = f"<{self.EXP_ID_translation[self.ORIG_ID]}> \n {category} parameter ranges"
#         from highlight_text import fig_text
#         fig_text(
#             x=0.5, # position on x-axis (centered)
#             y=0.96, # position on y-axis
#             s=text,
#             fontsize=12,
#             ha='center',
#             textalign = 'center',
#             highlight_textprops=[{'color': fontcol, 'fontweight': 'bold'}]
#         )
#         # f1.suptitle(f'{self.EXP_ID_translation[self.ORIG_ID]} \n {category} parameter ranges', 
#         #             fontsize=12, y=0.96)
#         f1.savefig(join(self.FIGDIR, f'{category}_bothparamkinds_parameters.png'),
#                     bbox_inches='tight', dpi=300)

#         # plt.close(f1)
#     all_years_param_plot(melted_params,category = 'Posterior')




#%%
#plot SWE and Q for particular year 
# for SWE_chosen in SWE_targets:
for self in LOA_objects.values():
    if 'LB' in self.ORIG_ID:
        continue
    SWE_PRI_dic = self.SWE_PRI_dic
    
    for SWE_chosen in ['melt_sum_grid_MAPE']:
    # for SWE_chosen in ['t_SWE_end_elev_ME']:
        SEL = SWE_PRI_dic[SWE_chosen]['SWE_results'][SWE_PRI_dic[SWE_chosen]['SWE_results']['Q_metric'] == Q_chosen]
        yearlyfigdir = join(self.FIGDIR, 'allyears')
        Path(yearlyfigdir).mkdir(parents=True, exist_ok=True)
        for year in [2003]:
            
            # yearlyfigdir = join(self.FIGDIR, str(year))
            # Path(yearlyfigdir).mkdir(parents=True, exist_ok=True)
        # for year in range(self.START_YEAR, self.END_YEAR+1):
            all_ids = SEL[SEL['year'] == year]['index']
            post_ids = SEL[(SEL['category'] == 'Posterior') & (SEL['year'] == year)]['index'].values
            UB_id = SEL[(SEL['category'] == 'UB') & (SEL['year'] == year)]['index'].values
            # LB_id = SEL[(SEL['category'] == 'LB') & (SEL['year'] == year)]['index'].values


            #SWE plot
            SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
            SWE_sims_all = self.SWE[year]
            SWE_sims_post = SWE_sims_all[post_ids]
            SWE_sims_UB = SWE_sims_all[UB_id]
            # SWE_sims_LB = Lower_Benchmark.SWE[year][LB_id]

            SWEobs_bands = self.E.swe2bands(SWEobs,bands = list(self.elev_bands))
            SWE_sims_all_bands = self.E.swe2bands(SWE_sims_all,bands = list(self.elev_bands))
            SWEsims_good_bands = self.E.swe2bands(SWE_sims_post,bands = list(self.elev_bands))
            upper_benchmark_bands = self.E.swe2bands(SWE_sims_UB,bands = list(self.elev_bands))
            # lower_benchmark_bands = Lower_Benchmark.E.swe2bands(SWE_sims_LB,bands = list(self.elev_bands))
            bands  = list(SWEobs_bands.keys())

            f1, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 5))
            for i, band in enumerate(bands):
                ax1 = axes[i]
                obsplot = SWEobs_bands[band].plot(ax=ax1, color='black', 
                                                    label='SWEobs', zorder=100,
                                                    linestyle = '--')
                SWE_all_t = SWE_sims_all_bands[band].drop(columns='spatial_ref', errors='ignore')
                goodsims = SWEsims_good_bands[band].drop(columns='spatial_ref', errors='ignore')
                simplot = goodsims.plot(ax=ax1, color=colors['Posterior'], label='_noLegend', legend=False,
                                        alpha=alphas['Posterior'], linestyle='-')
                UB_bands = upper_benchmark_bands[band].drop(columns='spatial_ref', errors='ignore')
                UB_bands.plot(ax=ax1, color=colors['UB'], alpha = alphas['UB'],
                                linestyle='-',label = '_noLegend')
                # LB_bands = lower_benchmark_bands[band].drop(columns='spatial_ref', errors='ignore')
                # fill_plot_LB = ax1.fill_between(SWE_all_t.index, LB_bands.min(axis=1), LB_bands.max(axis=1), 
                #                                 color=colors['LB'], alpha=0.2, zorder=0)

                # ax1.legend(handles=linehandles, labels=['SWEobs', 'SWEsim_selection', 'All SWEsim', 'Upper benchmark', 'Lower benchmark'])
                ax1.legend([])
                ax1.set_title(f"{band}")
                ax1.grid()
            plt.savefig(join(yearlyfigdir, f"{year}_SWE_bands__{SWE_chosen}.png"), dpi=300, bbox_inches='tight')
            # plt.show()

            # handles = [Line2D([0], [0], color='black', linestyle='--')]
            handles = [linehandle_obs,linehandles['Posterior'], linehandles['UB']]#, patch_handles['LB']]
            labels = ['Observed', 'Posterior', 'Upper benchmark']#, 'Lower benchmark']
            ax1.legend(handles, labels)
            ax1.set_title(f"{band}")
            plt.show()

            #SWE output plot 
            #calculate the net SWE decrease for each time step over the catchment 
            def swe3d_to_melt(swe3d):
                swe1d = swe3d.sum(dim = ['lat','lon'])
                swe3d_melt = swe1d.diff('time')
                swe3d_melt = xr.where(swe3d_melt > 0, 0, swe3d_melt)*-1
                melt1d = swe3d_melt.to_pandas()
                if isinstance(melt1d, pd.DataFrame) and 'spatial_ref' in melt1d.columns:
                    melt1d = melt1d.drop(columns = 'spatial_ref')
                return melt1d
            SWEobs_melt = swe3d_to_melt(SWEobs)
            SWE_sims_all_melt = swe3d_to_melt(SWE_sims_all)
            SWE_sims_post_melt = swe3d_to_melt(SWE_sims_post)
            SWE_sims_UB_melt = swe3d_to_melt(SWE_sims_UB)
            # SWE_sims_LB_melt = swe3d_to_melt(SWE_sims_LB)

            def swe3d_to_melt_area(swe3d):
                swe3d_melt = swe3d.diff(dim = 'time')
                swe3d_melt = xr.where(swe3d_melt >= 0, 0, swe3d_melt)*-1
                melt_area = xr.where(swe3d_melt > 0, 1, 0).sum(dim = ['lat','lon'])
                melt1d = melt_area.to_pandas()
                if isinstance(melt1d, pd.DataFrame) and 'spatial_ref' in melt1d.columns:
                    melt1d = melt1d.drop(columns = 'spatial_ref')
                return melt1d
            SWEobs_melt_area = swe3d_to_melt_area(SWEobs)
            SWE_sims_all_melt_area = swe3d_to_melt_area(SWE_sims_all)
            SWE_sims_post_melt_area = swe3d_to_melt_area(SWE_sims_post)
            SWE_sims_UB_melt_area = swe3d_to_melt_area(SWE_sims_UB)
            # SWE_sims_LB_melt_area = swe3d_to_melt_area(SWE_sims_LB)

            specmelt_obs = SWEobs_melt.div(SWEobs_melt_area)
            specmelt_all = SWE_sims_all_melt.div(SWE_sims_all_melt_area)
            specmelt_post = SWE_sims_post_melt.div(SWE_sims_post_melt_area)
            specmelt_UB = SWE_sims_UB_melt.div(SWE_sims_UB_melt_area)
            # specmelt_LB = SWE_sims_LB_melt.div(SWE_sims_LB_melt_area)



            f1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6),sharex=True)
            handles = [dothandle_obs, linehandle_obs, linehandles['Posterior'], linehandles['UB']]#, patchhandle_obs]
            labels = ['SWEobs', 'Posterior', 'Upper benchmark']#, 'Lower benchmark']
            # Plot total melt
            SWEobs_melt.plot(ax=ax1, color='black', label='SWEobs', linestyle='--',zorder=102)
            SWE_sims_post_melt.plot(ax=ax1, color=colors['Posterior'], alpha=alphas['Posterior'], label='_noLegend', legend=None)
            SWE_sims_UB_melt.plot(ax=ax1, color=colors['UB'], linestyle='--', label='Upper benchmark',zorder = 1)
            # ax1.fill_between(SWE_sims_all_melt.index, SWE_sims_LB_melt.min(axis=1), SWE_sims_LB_melt.max(axis=1), color=colors['LB'], alpha=0.2)
            ax1.set_xlim(f"{year}-03-01", f"{year}-07-30")
            ax1.set_title('Total Meltwater Production')
            ax1.set_ylabel('SWE dz')
            ax1.grid()
            ax1.legend([])

            specmelt_obs.plot(ax=ax2, color='black', linestyle = '--',label='SWEobs',zorder = 102)
            specmelt_post.plot(ax=ax2, color=colors['Posterior'], alpha=alphas['Posterior'], label='Posterior')
            specmelt_UB.plot(ax=ax2, color=colors['UB'], linestyle='--', alpha=alphas['UB'], zorder=1)
            # ax2.fill_between(SWE_sims_all_melt.index, specmelt_LB.min(axis=1).astype(float), specmelt_LB.max(axis=1).astype(float), color=colors['LB'], alpha=0.2)
            ax2.set_xlim(f"{year}-03-01", f"{year}-07-30")
            ax2.set_title('Specific Meltwater Production')
            ax2.set_ylabel('SWE dz')
            ax2.grid()
            ax2.set_xlabel(None)
            ax2.legend(handles=[linehandle_obs, linehandles['Posterior'], linehandles['UB'], patch_handles['LB']],
                    labels = labels)
            plt.tight_layout()
            plt.show()  

            #Plot cumulative melt 
            #first over entire catchment 
            # f1,ax1 = plt.subplots(figsize = (5,5))
            # SWEobs_melt.cumsum().plot(ax = ax1, color = 'black', label = 'Observed',zorder = 100)
            # SWE_sims_all_melt.cumsum().plot(ax=ax1, color=colors['Prior'], label='All Simulations', alpha=0.5)
            # SWE_sims_post_melt.cumsum().plot(ax=ax1, color=colors['Posterior'], label='Posterior', alpha=0.5)
            # # SWE_sims_UB_melt.plot(ax=ax1, color=colors['UB'], linestyle='--', label='Upper Benchmark')
            # # SWE_sims_LB_melt.plot(ax=ax1, color=colors['LB'], linestyle='--', label='Lower Benchmark')
            # ax1.set_title('Cumulative Meltwater Comparison')
            # ax1.set_ylabel('Cumulative Melt (SWE dz)')
            # ax1.legend()
            # ax1.grid()

            #now over elevation bands
            # f1,ax1 = plt.subplots(figsize = (5,5))

            #Q plot 
            timeslice = slice(f'{year}-02-01',f'{year}-09-30')
            QQall = self.Q.loc[timeslice]
            QQmin = QQall.min(axis = 1)
            QQmax = QQall.max(axis = 1)

            QQpost = self.Q.loc[timeslice][post_ids]
            QQobs = self.Qobs.loc[timeslice]
            QQ_UB = SWE_PRI_dic[SWE_chosen]['Q_UB'][year][timeslice]
            # QQ_UB = self.Q.loc[timeslice][UB_id]

            f1,ax1 = plt.subplots(figsize = (7,4))
            try:
                QQpost.plot(ax = ax1, color = colors['Posterior'],alpha = alphas['Posterior'],legend = None,
                            zorder = 50)
            except:
                print(f"Year {year} has no posterior runs")
                pass
            QQobs.plot(ax = ax1,color= 'black', linestyle = '--', legend = None,zorder = 100)
            QQ_UB.plot(ax = ax1, color = colors['UB'],alpha = alphas['UB'], linestyle = '-', legend = None)
            ax1.fill_between(QQall.index, QQmin, QQmax, color = colors['Prior'], alpha = 0.2)
            handles = [linehandle_obs,linehandles['Posterior'], linehandles['UB'], patch_handles['Prior']]
            labels = ['Observed', 'Posterior', 'Upper benchmark', 'Prior']
            # labels = ['Posterior','Obs', 'Upper benchmark', 'Prior']
            ax1.legend(handles, labels)
            ax1.grid()
            ax1.set_title('Streamflow')
            ax1.set_ylabel('Q [m3/s]')
            ax1.set_xlabel(None)
            plt.savefig(join(yearlyfigdir, f'{year}_{SWE_chosen}_Q.png'), dpi=200)


            #metric over elev plot 
            sig = 'melt_sum_elev'
            obs_elev = self.swe_signatures[year]['obs'][sig]
            prior_elev = self.swe_signatures[year]['sim'][sig].loc[:,all_ids]
            post_elev = self.swe_signatures[year]['sim'][sig].loc[:,post_ids]
            UB_elev = self.swe_signatures[year]['sim'][sig].loc[:,UB_id]
            # LB_elev = Lower_Benchmark.swe_signatures[year]['sim'][sig]

            #plot it like i plotted all before
            f1,ax1 = plt.subplots(figsize = (5,5))
            ax1.plot(obs_elev.values, obs_elev.index, color = 'black', linestyle = '--',
                        label = 'Observed',zorder = 100)
            ax1.plot(post_elev.values,post_elev.index, color = colors['Posterior'], 
                        alpha = alphas['Posterior'], label = 'Posterior',zorder = 50)
            ax1.plot(UB_elev.values,UB_elev.index, color = colors['UB'], 
                    linestyle = '--', label = 'Upper benchmark',zorder = 90)
            ax1.fill_betweenx(UB_elev.index, prior_elev.min(axis = 1).astype(float),
                            prior_elev.max(axis = 1).astype(float), color = colors['Prior'], alpha = 0.2)
            ax1.plot(prior_elev.median(axis = 1).astype(float), prior_elev.index, color = 'grey', 
                    zorder = 200)
            # ax1.fill_betweenx(UB_elev.index, LB_elev.min(axis = 1).astype(float), 
            #                 LB_elev.max(axis = 1).astype(float), color = colors['LB'], alpha = 0.2)
            ax1.set_title('SWE Elevation Comparison')
            ax1.set_xlabel(sig)
            ax1.set_ylabel('Elevation Band')
            handles = [linehandle_obs, patch_handles['Prior'], linehandles['Posterior'], linehandles['UB']]#, patch_handles['LB']]
            labels = ['Observed', 'Prior','Posterior', 'Upper benchmark']#, 'Lower benchmark']
            ax1.legend(handles, labels)
            ax1.grid()
            plt.savefig(join(yearlyfigdir, f'{year}_{SWE_chosen}_melt_sum_elev.png'), dpi=200)
            plt.show()

    #Forcing error exploration 
    #start with snowfall - spatial 
    #SWE_sims_UB, SWE_sims_UB_melt, UB_id 
    
  
    ##%%
    def calc_cum_sf_grid(swe3d):
        diff = swe3d.diff(dim = 'time')
        diff = xr.where(diff < 0, 0, diff)
        # cum_sf = diff.cumsum(dim = 'time')
        cum_sf = diff 
        return cum_sf
    def calc_cum_melt_grid(swe3d):
        diff = swe3d.diff(dim = 'time')
        diff = xr.where(diff > 0, 0, diff)*-1
        cum_melt = diff.cumsum(dim = 'time')
        cum_melt = diff
        return cum_melt
    def calc_cum_error(obs1dcum,sim1dcum):
        # err1d = np.abs(obs1dcum - sim1dcum)
        # absmax = np.nanmax(err1d)
        if np.isnan(obs1dcum).all() or np.isnan(sim1dcum).all():
            absmax = np.nan
        else:
            absmax = he.mae(obs1dcum, sim1dcum)
        # if err1d.any():
        #     plt.figure()
        #     plt.plot(obs1dcum, label='Observed')
        #     plt.plot(sim1dcum, label='Simulated')
        #     plt.axvline(np.argmax(err1d), color='black', linestyle='--')
        #     plt.legend()
        #     plt.title(f"Max error = {absmax}")
        #     plt.show()
        return absmax
    def swe3d_to_melt3d(swe3d):
        diff = swe3d.diff(dim = 'time')
        diff = xr.where(diff > 0, 0, diff)*-1
        return diff
    def swe3d_to_sf3d(swe3d):
        diff = swe3d.diff(dim = 'time')
        diff = xr.where(diff < 0, 0, diff)
        return diff 


    

    # diff1d = SWEobs.mean(dim = ['lat','lon']).diff(dim = 'time')
    # diff1d = xr.where(diff1d > 0, 0, diff1d)*-1
    # cum_melt = diff1d.cumsum(dim = 'time')

    spatial_metrics = pd.DataFrame( columns= ['SPAEF',Q_chosen,'year'])
    # spatial_melt = pd.DataFrame(index = UB_id, columns= ['SPAEF',Q_chosen,'year'])
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        all_ids = SEL[SEL['year'] == year]['index']
        post_ids = SEL[(SEL['category'] == 'Posterior') & (SEL['year'] == year)]['index'].values
        UB_id = SEL[(SEL['category'] == 'UB') & (SEL['year'] == year)]['index'].values
        LB_id = SEL[(SEL['category'] == 'LB') & (SEL['year'] == year)]['index'].values

        #SWE plot
        SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
        SWE_sims_all = self.SWE[year]
        SWE_sims_post = SWE_sims_all[post_ids]
        SWE_sims_UB = SWE_sims_all[UB_id].sel(time = SWEobs.time)

        # obs2d = SWEobs.sum(dim = 'time', skipna = True)
        # obs2d = xr.where(obs2d ==0, np.nan, obs2d)
        obs_melt_2d = self.calc_melt_sum_grid(SWEobs)
        obs_cum_melt_grid = calc_cum_melt_grid(SWEobs)
        obs_cum_sf_grid = calc_cum_sf_grid(SWEobs)
        obs_cum_melt_catchment = obs_cum_melt_grid.mean(dim = ['lat','lon'])
        obs_cum_sf_catchment = obs_cum_sf_grid.mean(dim = ['lat','lon'])

        # f1,(ax1,ax2) = plt.subplots(2,1,figsize = (5,10))
        # obs_cum_melt_catchment.plot(ax = ax1,color = 'black',linestyle = '--',zorder = 100)
        # obs_cum_sf_catchment.plot(ax = ax2, color = 'black', linestyle = '--', zorder = 100)

        # point = [5,5]
        # f1,(ax3,ax4) = plt.subplots(2,1,figsize = (5,10))
        # obs_cum_melt_grid.isel(lat = point[0], lon = point[1]).plot(ax = ax3, color = 'black', linestyle = '--', zorder = 100)
        # obs_cum_sf_grid.isel(lat = point[0], lon = point[1]).plot(ax = ax4, color = 'black', linestyle = '--', zorder = 100)
        for id in UB_id: 
            # sim2d = SWE_sims_UB[id].sum(dim = 'time',skipna = True)
            # sim2d = xr.where(sim2d ==0, np.nan, sim2d)
            
            # spaef = calculate_spaef(obs2d, sim2d)
            # spatial_metrics.loc[id, 'SPAEF_sf'] = spaef
            # spatial_metrics.loc[id, Q_chosen] = self.metrics[year][Q_chosen].loc[id]
            Q_metrics = SWE_PRI_dic[SWE_chosen]['Q_results']
            spatial_metrics.loc[id, Q_chosen] = Q_metrics[Q_metrics['year'] == year][Q_metrics['category'] == 'UB'][Q_metrics['index'] == id][Q_chosen].values[0]
            spatial_metrics.loc[id, 'year'] = year

            sim_melt = self.calc_melt_sum_grid(SWE_sims_UB[id])
            spaef_melt = self.calculate_spaef(SWEobs, SWE_sims_UB[id])
            spatial_metrics.loc[id, 'SPAEF'] = spaef_melt

            sim_cum_melt_grid = calc_cum_melt_grid(SWE_sims_UB[id])
            sim_cum_sf_grid = calc_cum_sf_grid(SWE_sims_UB[id])

            sim_cum_melt_catchment = sim_cum_melt_grid.mean(dim = ['lat','lon'])
            sim_cum_sf_catchment = sim_cum_sf_grid.mean(dim = ['lat','lon'])

            temp_error_melt_grid = xr.apply_ufunc(calc_cum_error, obs_cum_melt_grid, sim_cum_melt_grid, input_core_dims=[['time'],['time']],
                                    output_core_dims=[[]], vectorize = True)
            temp_error_melt_grid = xr.where(temp_error_melt_grid == 0, np.nan, temp_error_melt_grid)
            temp_error_sf_grid = xr.apply_ufunc(calc_cum_error, obs_cum_sf_grid, sim_cum_sf_grid, input_core_dims=[['time'],['time']],
                                    output_core_dims=[[]], vectorize = True)
            temp_error_sf_grid = xr.where(temp_error_sf_grid == 0, np.nan, temp_error_sf_grid)
            temp_error_sf_catchment = calc_cum_error(obs_cum_sf_catchment, sim_cum_sf_catchment)
            temp_error_melt_catchment = calc_cum_error(obs_cum_melt_catchment, sim_cum_melt_catchment)

            spatial_metrics.loc[id, 'temp_error_melt_grid'] = temp_error_melt_grid.mean()
            spatial_metrics.loc[id, 'temp_error_sf_grid'] = temp_error_sf_grid.mean()
            spatial_metrics.loc[id, 'temp_error_sf_catchment'] = temp_error_sf_catchment
            spatial_metrics.loc[id, 'temp_error_melt_catchment'] = temp_error_melt_catchment

            # sim_cum_melt_catchment.plot(ax = ax1, color = colors['UB'], linestyle = '-')
            # sim_cum_sf_catchment.plot(ax = ax2, color = colors['UB'], linestyle = '-')

            # sim_cum_melt_grid.isel(lat = point[0], lon = point[1]).plot(ax = ax3, color = colors['UB'], linestyle = '-')
            # sim_cum_sf_grid.isel(lat = point[0], lon = point[1]).plot(ax = ax4, color = colors['UB'], linestyle = '-')
        # handles = [linehandle_obs, linehandles['UB']]
        # labels = ['Observed', 'Upper Benchmark']
        # ax1.legend(handles, labels)
        # ax1.set_title('Cumulative Catchment Meltwater Production')
        # ax1.set_ylabel('mm')
        # ax2.set_title('Cumulative Catchment Snowfall Production')
        # ax2.set_ylabel('mm')
        # plt.show()

        # ax3.legend(handles, labels)
        # ax3.set_title('Cumulative Grid Meltwater Production')
        # ax3.set_ylabel('mm')
        # ax4.set_title('Cumulative Grid Snowfall Production')
        # ax4.set_ylabel('mm')
        # plt.show()

    spatial_metrics = spatial_metrics.astype(float).reset_index()
    # plt.figure()
    # for year in range(self.START_YEAR, self.END_YEAR + 1):
    #     year_data = spatial_metrics[spatial_metrics['year'] == year]
    #     sns.scatterplot(data = year_data, x = 'SPAEF', y = Q_chosen,hue = 'year')
        # sns.regplot(data = year_data, x = 'SPAEF', y = Q_chosen, scatter = False)
    def reg_and_scatter(x,y,data,ax, legend = True):
        colors = plt.cm.viridis(np.linspace(0,1,len(data['year'].unique())))
        # plt.figure()
        r2_values = []
        for i, year in enumerate(range(self.START_YEAR, self.END_YEAR + 1)):
            year_data = data[data['year'] == year]
            sns.scatterplot(data=year_data, x=x, y=y, color='black',alpha = 0.4, ax = ax)
            sns.regplot(data=year_data, x=x, y=y, scatter=False, color=colors[i], 
                        label=int(year), ax= ax)
            r2 = np.corrcoef(year_data[x], year_data[y])[0, 1]**2
            r2_values.append(r2)
        if legend:
            ax.legend(ncols = 2,loc = [1.01,0.2])
        else:
            ax.legend([])
        ax.set_title(f"Mean R² = {np.median(r2_values):.2f}")
        # plt.tight_layout()

    f1,axes = plt.subplots(1,3,figsize = (15,5),sharey=True)
    reg_and_scatter('SPAEF',Q_chosen,spatial_metrics,axes[0], legend = False)
    reg_and_scatter('temp_error_sf_grid',Q_chosen,spatial_metrics,axes[1],legend = False)
    reg_and_scatter('temp_error_melt_grid',Q_chosen,spatial_metrics,axes[2])
    axes[0].set_xlabel('SPAEF [-]')
    axes[0].set_xlim(right = 1)
    axes[1].set_xlabel('Snowfall mean timing error [mm]')
    axes[2].set_xlabel('Melt mean timing error [mm]')
    axes[1].set_xlim(left = 0)
    axes[2].set_xlim(left = 0)
    for ax in axes:
        ax.set_ylim(0,1)
        ax.grid()
    plt.tight_layout()

    # f1,axes = plt.subplots(1,3,figsize = (15,5))
    # reg_and_scatter('SPAEF',Q_chosen,spatial_metrics,axes[0], legend = False)
    # reg_and_scatter('temp_error_sf_catchment',Q_chosen,spatial_metrics,axes[1],legend = False)
    # reg_and_scatter('temp_error_melt_catchment',Q_chosen,spatial_metrics,axes[2])
    ##%%
    # Calc outside prior range error for all runs 
    def calc_outside_prior_err(obs1d,sim1d_min,sim1d_max,plot = False):
        # if isinstance(obs1d, xr.DataArray):
        #     obs1d = obs1d.to_pandas()
        err_above = (obs1d < sim1d_min) * (sim1d_min - obs1d)
        err_below = (obs1d > sim1d_max) * (obs1d - sim1d_max)
        outside_prior = err_above + err_below
        if outside_prior.sum()>300: 
            plot=True
        if plot:
            plt.figure(figsize = (5,2))
            plt.plot(obs1d, color = 'black', 
                        label = 'Observed', linestyle = '--')
            plt.fill_between(np.arange(len(sim1d_min)), 
                                sim1d_min, sim1d_max, 
                                color = 'tab:blue', alpha = 0.5,
                                label = 'All runs range')
            plt.legend()
            plt.ylabel('Melt (mm/day)')
            plt.show()
        return outside_prior.sum()
    err_df = pd.DataFrame(columns = ['year','error'])
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        print(year)
        all_ids = SEL[SEL['year'] == year]['index']
        SWEobs = self.SWEobs.sel(time = slice(f'{year-1}-10-01',f'{year}-09-30'))
        SWE_sims_all = self.SWE[year].sel(time = SWEobs.time)
        SWEobs_melt = swe3d_to_melt3d(SWEobs)
        SWE_sims_all_melt = swe3d_to_melt3d(SWE_sims_all)
        SWE_sims_all_melt_min = SWE_sims_all_melt.to_array().min(dim = 'variable')
        SWE_sims_all_melt_max = SWE_sims_all_melt.to_array().max(dim = 'variable')
        grid_error = xr.apply_ufunc(calc_outside_prior_err, SWEobs_melt, 
                                    SWE_sims_all_melt_min, SWE_sims_all_melt_max, 
                                    input_core_dims=[['time'],['time'],['time']],
                                    output_core_dims=[[]], vectorize = True)
        catchment_error = grid_error.sum().item()
        err_df.loc[year, 'error'] = catchment_error
        err_df.loc[year, 'year'] = year
    plt.figure(figsize = (5,3))
    err_df['error'].plot(kind = 'bar',
                            title='Uncorrectable melt error'
                            , ylabel='Melt error [mm/day * cell]', xlabel='Year')
    plt.tight_layout()
    plt.show()

    # Sort error dataframe
    err_df_sorted = err_df.sort_values('error', ascending=True)
    sorted_years = err_df_sorted.index.tolist()

    # Process SWE dataframe
    rank_stack_swe['Year'] = pd.Categorical(rank_stack_swe['Year'], categories=sorted_years, ordered=True)
    rank_stack_swe = rank_stack_swe.sort_values('Year')
    rank_stack_post = rank_stack_swe[rank_stack_swe['Category'] == 'Posterior'].copy()

    # Process Q dataframe
    rank_stack['Year'] = pd.Categorical(rank_stack['Year'], categories=sorted_years, ordered=True)
    rank_stack = rank_stack.sort_values('Year')
    rank_stack_Q_post = rank_stack[rank_stack['Category'] == 'UB'].copy()

    # Add melt error to SWE dataframe
    rank_stack_post['Melt_error'] = np.nan
    for year in sorted_years:
        catchment_error = err_df_sorted.loc[year, 'error']
        rank_stack_post.loc[rank_stack_post['Year'] == year, 'Melt_error'] = catchment_error
    rank_stack_post['Melt_error'] = rank_stack_post['Melt_error'].astype(float)
    rank_stack_post.drop('Category', axis=1, inplace=True)
    rank_stack_post = rank_stack_post.astype(float)

    # Add melt error to Q dataframe
    rank_stack_Q_post['Melt_error'] = np.nan
    for year in sorted_years:
        catchment_error = err_df_sorted.loc[year, 'error']
        rank_stack_Q_post.loc[rank_stack_Q_post['Year'] == year, 'Melt_error'] = catchment_error
    rank_stack_Q_post['Melt_error'] = rank_stack_Q_post['Melt_error'].astype(float)
    rank_stack_Q_post.drop('Category', axis=1, inplace=True) 
    rank_stack_Q_post = rank_stack_Q_post.astype(float)


    # Create a figure with two subplots: top for regression, bottom for boxplots
    fig, ax1 = plt.subplots(1, 1, figsize=(5,4), sharex=True, 
                                )

    # Top subplot: Regression plot
    sns.regplot(data=rank_stack_post, x='Melt_error', y='Rank', 
                color=colors['Posterior'], scatter=True, line_kws={'alpha':0.8},
                scatter_kws={'alpha':0.05}, ax=ax1)

    # Calculate regression statistics
    from scipy.stats import linregress
    x = rank_stack_post['Melt_error'].values
    y = rank_stack_post['Rank'].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    ax1.set_title('Regression Analysis')
    ax1.annotate(f"r = {r_value:.2f}, p = {p_value:.2f}\ny = {slope:.2f}x + {intercept:.2f}", 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('SWE Rank')

    # Bottom subplot: Boxplot analysis
    # Get the x-tick positions from the regression plot
    xticks = ax1.get_xticks()
    xtick_labels = ax1.get_xticklabels()

    # Group data by Melt_error values
    unique_errors = rank_stack_post['Melt_error'].unique()
    unique_errors.sort()

    # Collect ranks for each unique melt error
    box_data = []
    positions = []
    for error in unique_errors:
        ranks = rank_stack_post[rank_stack_post['Melt_error'] == error]['Rank'].values
        if len(ranks) > 0:  # Only add if there are points for this error value
            box_data.append(ranks)
            positions.append(error)

    # Create boxplots at the exact melt_error positions
    bp = ax1.boxplot(box_data, positions=positions, widths=min(0.01*max(positions), 10000), 
                    patch_artist=True, showfliers=False)

    # Style the boxplots to match the Posterior color with transparency
    for box in bp['boxes']:
        box.set(color=colors['Posterior'], alpha=0.6)
        box.set(facecolor=colors['Posterior'], alpha=0.3)
    for whisker in bp['whiskers']:
        whisker.set(color=colors['Posterior'], alpha=0.6)
    for cap in bp['caps']:
        cap.set(color=colors['Posterior'], alpha=0.6)
    for median in bp['medians']:
        median.set(color=colors['Posterior'], linewidth=2)

    # Add a regression line to the boxplot for reference
    x_range = np.linspace(min(positions), max(positions), 100)
    ax1.plot(x_range, slope * x_range + intercept, 'r--', alpha=0.7, linewidth=1.5)

    ax1.set_title('Relationship between Uncorrectable Melt Error and SWE Rank')
    ax1.set_xlabel('Structural error [mm/day * cell]')
    ax1.set_ylabel('SWE Rank')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)

    # Add overall title
    # fig.suptitle('Relationship between Melt Error and SWE Rank', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.show()

    #same for rank_stack_Q_post
    fig, ax1 = plt.subplots(1, 1, figsize=(5,4), sharex=True, 
                                )

    # Top subplot: Regression plot
    sns.regplot(data=rank_stack_Q_post, x='Melt_error', y='Rank', 
                color=colors['UB'], scatter=True, line_kws={'alpha':0.8},
                scatter_kws={'alpha':0.05}, ax=ax1)

    # Calculate regression statistics
    x = rank_stack_Q_post['Melt_error'].values
    y = rank_stack_Q_post['Rank'].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    ax1.set_title('Regression Analysis')
    ax1.annotate(f"r = {r_value:.2f}, p = {p_value:.2f}\ny = {slope:.2f}x + {intercept:.2f}", 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('Q Rank')

    # Bottom subplot: Boxplot analysis
    # Get the x-tick positions from the regression plot
    xticks = ax1.get_xticks()
    xtick_labels = ax1.get_xticklabels()

    # Group data by Melt_error values
    unique_errors = rank_stack_Q_post['Melt_error'].unique()
    unique_errors.sort()

    # Collect ranks for each unique melt error
    box_data = []
    positions = []
    for error in unique_errors:
        ranks = rank_stack_Q_post[rank_stack_Q_post['Melt_error'] == error]['Rank'].values
        if len(ranks) > 0:  # Only add if there are points for this error value
            box_data.append(ranks)
            positions.append(error)

    # Create boxplots at the exact melt_error positions
    bp = ax1.boxplot(box_data, positions=positions, widths=min(0.01*max(positions), 10000), 
                    patch_artist=True, showfliers=False)

    # Style the boxplots to match the UB color with transparency
    for box in bp['boxes']:
        box.set(color=colors['UB'], alpha=0.6)
        box.set(facecolor=colors['UB'], alpha=0.3)
    for whisker in bp['whiskers']:
        whisker.set(color=colors['UB'], alpha=0.6)
    for cap in bp['caps']:
        cap.set(color=colors['UB'], alpha=0.6)
    for median in bp['medians']:
        median.set(color=colors['UB'], linewidth=2)

    # Add a regression line to the boxplot for reference
    x_range = np.linspace(min(positions), max(positions), 100)
    ax1.plot(x_range, slope * x_range + intercept, 'r--', alpha=0.7, linewidth=1.5)

    ax1.set_title('Relationship between Uncorrectable Melt Error and Q Rank')
    ax1.set_xlabel('Structural error [mm/day * cell]')
    ax1.set_ylabel('Q Rank')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)

    # Add overall title
    # fig.suptitle('Relationship between Melt Error and SWE Rank', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.show()

    # Plot 1: SWE vs Melt error (orange)
    # plt.figure(figsize=(7,4))
    # sns.regplot(data=rank_stack_post, x='Melt_error', y='Rank', 
    #             color=colors['Posterior'], scatter_kws={'alpha':0.6})
    # # sns.pointplot(data=rank_stack_post, x='Melt_error', y='Rank',
    # #             color=colors['Posterior'], join=False, ci=None)
    
    # from scipy.stats import linregress
    # x = rank_stack_post['Melt_error'].values
    # y = rank_stack_post['Rank'].values
    # slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # plt.title('Relationship between Melt Error and SWE Rank')
    # plt.xlabel('Melt Error [mm/day * cell]')
    # plt.ylabel('SWE Rank')
    # plt.annotate(f"r = {r_value:.2f}, p = {p_value:.2f}\ny = {slope:.2f}x + {intercept:.2f}", 
    #             xy=(0.05, 0.95), xycoords='axes fraction', 
    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    # plt.tight_layout()
    # plt.show()

    # Plot 2: Q vs Melt error (green)
    # plt.figure(figsize=(7,4))
    # sns.regplot(data=rank_stack_Q_post, x='Melt_error', y='Rank', color=colors['UB'], scatter_kws={'alpha':0.6})

    # x = rank_stack_Q_post['Melt_error'].values
    # y = rank_stack_Q_post['Rank'].values
    # slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # plt.title('Relationship between Melt Error and Q Rank')
    # plt.xlabel('Melt Error [mm/day * cell]')
    # plt.ylabel('Q Rank')
    # plt.annotate(f"r = {r_value:.2f}, p = {p_value:.2f}\ny = {slope:.2f}x + {intercept:.2f}", 
    #             xy=(0.05, 0.95), xycoords='axes fraction',
    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    # plt.tight_layout()
    # plt.show()
    
    
    ##%% Check for all years when obs Q is outside all_runs range 
    Qobs = self.Qobs.squeeze()
    Qall = self.Q
    Qmin = Qall.min(axis = 1)
    Qmax = Qall.max(axis = 1)
    outside_prior = (Qobs < Qmin) | (Qobs > Qmax)
    yearly_outside_sum = outside_prior.resample('AS-OCT').sum()
    plt.figure()
    yearly_outside_sum.index = yearly_outside_sum.index.year+1
    yearly_outside_sum.plot(kind='bar', title='Number of days where Obs is outside prior range',
                                xlabel='Year', ylabel='Count Outside Range')
    #recalculate KGE_meltseason, but only on the days where obs is within prior rang
    kge_filtered = pd.DataFrame(index = Qall.columns, columns = range(self.START_YEAR, self.END_YEAR + 1))
    nse_filtered = pd.DataFrame(index = Qall.columns, columns = range(self.START_YEAR, self.END_YEAR + 1))
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        timeslice = slice(f'{year-1}-04-01',f'{year}-07-30')
        Qobs_filtered = Qobs.loc[timeslice].where(~outside_prior.loc[timeslice])
        Qsim_filtered = Qall.loc[timeslice].where(~outside_prior.loc[timeslice])
        for run_id in Qall.columns:
            kge_filtered.loc[run_id, year] = he.kge_2009(Qobs_filtered, Qsim_filtered[run_id], return_all=False)
    
            nse_filtered.loc[run_id, year] = he.nse(Qobs_filtered, Qsim_filtered[run_id])
            
    kge_orig = pd.concat([self.metrics[year]['KGE_meltseason'] for year in range(self.START_YEAR, self.END_YEAR + 1)], axis=1)
    kge_orig.columns = range(self.START_YEAR, self.END_YEAR + 1)
    nse_orig = pd.concat([self.metrics[year]['NSE_meltseason'] for year in range(self.START_YEAR, self.END_YEAR + 1)], axis=1)
    nse_orig.columns = range(self.START_YEAR, self.END_YEAR + 1)

    #compare the two
    kge_orig_stack = kge_orig.stack().reset_index()
    kge_orig_stack.columns = ['run_id','year','KGE']
    kge_filtered_stack = kge_filtered.stack().reset_index()
    kge_filtered_stack.columns = ['run_id','year','KGE']
    kge_orig_stack['type'] = 'Original'
    kge_filtered_stack['type'] = 'Filtered'
    combined = pd.concat([kge_orig_stack,kge_filtered_stack])
    plt.figure()
    sns.boxplot(data = combined, x = 'year', y = 'KGE', hue = 'type')
    plt.title('KGE comparison between original and filtered data')
    plt.ylim(0,1)
    plt.show()

    #compare NSE 
    nse_orig_stack = nse_orig.stack().reset_index()
    nse_orig_stack.columns = ['run_id','year','NSE']
    nse_filtered_stack = nse_filtered.stack().reset_index()
    nse_filtered_stack.columns = ['run_id','year','NSE']
    nse_orig_stack['type'] = 'Original'
    nse_filtered_stack['type'] = 'Filtered'
    combined = pd.concat([nse_orig_stack,nse_filtered_stack])
    plt.figure()
    sns.boxplot(data = combined, x = 'year', y = 'NSE', hue = 'type')
    plt.title('NSE comparison between original and filtered data')
    plt.ylim(0,1)
    plt.show()

    #Calculate SWE_results with the new Filtered KGE 
    # WE first need to add the filtered_Kge to the metrics dictionary
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        self.metrics[year]['KGE_meltseason_filtered'] = kge_filtered[year].values
        self.metrics[year]['NSE_meltseason_filtered'] = nse_filtered[year].values
    
    #Now we can calculate the SWE_results
    Q_chosen_temp = ['NSE_meltseason_filtered']  # include NSE
    LOA_metrics_temp = self.calculate_LOA_metrics(Q_chosen_temp)  # use Q_chosen_temp instead of Q_chosen
    LOA_checks = self.perform_LOA_checks(Q_chosen_temp, LOA_metrics_temp)  # use Q_chosen_temp instead of Q_chosen
    SWE_results_filtered = self.gather_SWE_results(LOA_checks, Q_chosen, SWE_chosen,
                                            Lower_Benchmark=Lower_Benchmark)
    filtered_posterior = SWE_results_filtered[SWE_results_filtered['category'] == 'Posterior'].reset_index()
    orig_posterior = SEL[SEL['category'] == 'Posterior']
        
    #put them together in a melted dataframe 
    melted = pd.concat([orig_posterior,filtered_posterior], keys = ['Original NSE','Filtered NSE'])
    melted = melted.reset_index()
    melted = melted.drop('level_1', axis = 1)
    melted = melted.rename(columns = {'level_0':'type'})
    melted['year'] = melted['year'].astype(int)
    #plot the results
    plt.figure()
    sns.pointplot(data = melted, x = 'year', y = SWE_chosen, hue = 'type')
    plt.title("Calculating NSE only on days where Obs is within prior range \n does not lead to better SWE results")
    plt.xticks(rotation = 45)
    plt.ylabel(SWE_chosen)
    plt.grid()
    plt.show()

    # SWE_target_results_list.append(SWE_results)





#%%

    

    #     # ax1.get_legend().remove()





self.dotty_plot('melt_sum_APE','Qamp_APE')
# self.dotty_plot('melt_sum_ME','Qamp_ME')
# self.dotty_plot('melt_sum_elev_ME','Qmean_meltseason_ME')
# self.dotty_plot('melt_sum_elev_ME','Qmean_ME')
# self.dotty_plot('SWE_meltrate_elev_ME','t_hfd_ME')

# self.QSWEplot('Qmean_meltseason_ME')
self.QSWEplot('KGE_meltseason')
# self.QSWEplot('t_hfd_ME')
# self.QSWEplot('bfi_ME')






#%%
import matplotlib.pyplot as plt
import pandas as pd

def plot_correlations(LOA_objects, exp_id, swe_metric, q_metric):
    self = LOA_objects[exp_id]
    QSWE_corr = self.QSWE_corr.copy()

    # Filter QSWE_corr based on the provided SWE and Q metrics
    mask = QSWE_corr['level_0'].isin([swe_metric]) & QSWE_corr['level_1'].isin([q_metric])
    QSWE_corr = QSWE_corr[mask]

    # Extract Q signatures
    Qsignatures = self.Q_signatures[self.START_YEAR]['obs'].index
    Qsig_obs = pd.DataFrame(columns=Qsignatures, index=range(self.START_YEAR, self.END_YEAR + 1))
    for Qm in Qsignatures:
        Qsig_obs[Qm] = [self.Q_signatures[year]['obs'][Qm] for year in range(self.START_YEAR, self.END_YEAR + 1)]
    
    
    pr_sum_all = self.meteo_base['pr'].mean(dim = ['lat','lon'])
    #calculate yearly sum from October to september
    pr_sum_yearly = pr_sum_all.resample(time='AS-OCT').sum().to_pandas()[slice('2000','2021')]
    pr_sum_yearly.index = pr_sum_yearly.index.year+1
    Qsig_obs['Annual Precip'] = pr_sum_yearly

    pr_monthly = pr_sum_all.resample(time = 'M').sum().to_pandas()
    nov_to_may = pr_monthly.loc[pr_monthly.index.month.isin([11,12,1,2,3,4,5])]
    pr_sum_nov_to_may = nov_to_may.resample('AS-OCT').sum()[slice('2000','2021')]
    pr_sum_nov_to_may.index = pr_sum_nov_to_may.index.year+1
    Qsig_obs['Nov-May Precip'] = pr_sum_nov_to_may

    #tas mean
    tas_mean_all = self.meteo_base['tas'].mean(dim = ['lat','lon'])
    tas_mean_yearly = tas_mean_all.resample(time='AS-OCT').mean().to_pandas()[slice('2000','2021')]
    tas_mean_yearly.index = tas_mean_yearly.index.year + 1
    Qsig_obs['Annual Temperature'] = tas_mean_yearly

    #tas winter 
    tas_monthly = tas_mean_all.resample(time = 'M').mean().to_pandas()
    tas_winter = tas_monthly.loc[tas_monthly.index.month.isin([11,12,1,2,3,4,5])]
    tas_winter_mean = tas_winter.resample('AS-OCT').mean()[slice('2000','2021')]
    tas_winter_mean.index = tas_winter_mean.index.year + 1
    Qsig_obs['Winter Temperature'] = tas_winter_mean

    #snowfall fraction 
    # obs_snowfall = self.SWEobs.resample(time = 'AS-OCT').sum(dim = ['lat','lon']).to_pandas()
    
    snowfall_fractions = []
    for year in range(self.START_YEAR, self.END_YEAR+1):
        obs_snowfall = self.calc_melt_sum_catchment(self.SWEobs.sel(time = slice(f"{year-1}-10-01",f"{year}-09-30")))
        obs_precip = self.meteo_base['pr'].sel(time = slice(f"{year-1}-10-01",f"{year}-09-30")).mean(dim = ['lat','lon']).sum().item()
        snowfall_frac = obs_snowfall / obs_precip if obs_precip != 0 else 0
        snowfall_fractions.append(snowfall_frac)
        # print(f"Year {year} has snowfall = {obs_snowfall.sum()} with snowfall fraction = {snowfall_frac:.2f}")

    Qsig_obs['Snowfall fraction'] = snowfall_fractions
    QSWE_corr = QSWE_corr.set_index('Year')
    Qsig_obs['QSWE'] = QSWE_corr['correlation']

        #Calculation correlations with PRI 
    MD = SWE_PRI_dic[swe_metric]['PRI_SWE']
    MD = MD[MD['category'] == 'Posterior'].squeeze()
    MD.pop('category')
    MD = MD.set_index('year')

    Qsig_obs['MRD'] = MD['PRI'].values

    # Calculate correlations
    Qsig_QSWE_corr = Qsig_obs.corr('pearson')
    df = Qsig_QSWE_corr.loc['QSWE']
    df = df.drop('QSWE')
    df = df.sort_values(ascending=False)
    print(df)

    df2 = Qsig_QSWE_corr.loc['MRD']
    df2 = df2.drop('MRD')
    df2 = df2.sort_values(ascending=False)
    print(df2)

    df_frame = df.to_frame()
    df_frame = pd.concat([df_frame.head(5), df_frame.tail(5)])
    plt.figure(figsize = (2,4))
    sns.stripplot(data = df_frame, x = 'QSWE', y = df_frame.index)
    plt.grid()
    plt.title('Which Q+climate signatures predict \n successful Q-SWE inversion?')
    plt.xlabel('Correlation with SWE metric')
    plt.ylabel('Q + Climate Signatures')

    df2_frame = df2.to_frame()
    df2_frame = pd.concat([df2_frame.head(5), df2_frame.tail(5)])
    plt.figure(figsize = (2,4))
    sns.stripplot(data = df2_frame, x = 'MRD', y = df2_frame.index)
    plt.grid()
    plt.title('Which Q+climate signatures predict \n successful Q-SWE inversion?')
    plt.xlabel('Correlation with Median Rank Distance')
    plt.ylabel('Q + Climate Signatures')


    # Get the highest and lowest correlating Q signatures
    highest_corr = df.idxmax()
    highest_corr= df.index[1]
    lowest_corr = df.idxmin()

    highest_corr2 = df2.idxmax()
    highest_corr2 = df2.index[2]
    lowest_corr2 = df2.idxmin()

    plt.figure()

    
    plt.figure(figsize = (15,15))
    sns.heatmap(Qsig_obs.corr(), annot=False, cmap='coolwarm')
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))

    # Plot highest correlating Q signature
    years = range(self.START_YEAR, self.END_YEAR + 1)
    ax1.plot(years, QSWE_corr['correlation'].values, label=f'{swe_metric} - {q_metric} correlation')
    twin1 = ax1.twinx()
    twin1.plot(years, Qsig_obs[highest_corr].values, label=highest_corr, color='tab:orange')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Correlation')
    twin1.set_ylabel(highest_corr)
    ax1.legend(loc='upper left')
    twin1.legend(loc='upper right')
    ax1.set_title(f"Highest Correlation: {highest_corr} (Overall correlation = {df[highest_corr]:.2f})")

    # Plot lowest correlating Q signature
    ax2.plot(years, QSWE_corr['correlation'].values, label=f'{swe_metric} - {q_metric} correlation')
    twin2 = ax2.twinx()
    twin2.plot(years, Qsig_obs[lowest_corr].values, label=lowest_corr, color='orange')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Correlation')
    twin2.set_ylabel(lowest_corr)
    ax2.legend(loc='upper left')
    twin2.legend(loc='upper right')
    ax2.set_title(f"Lowest Correlation: {lowest_corr} (Overall correlation = {df[lowest_corr]:.2f})")

    fig.suptitle(f"Correlations between {swe_metric} and Q metrics")
    plt.tight_layout()
    plt.show()


    #plot against MRD
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
    ax1.plot(years, Qsig_obs['MRD'], label='Median Rank Distance')
    twin1 = ax1.twinx()
    twin1.plot(years, Qsig_obs[highest_corr2].values, label=highest_corr2, color='tab:orange')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('MRD')
    twin1.set_ylabel(highest_corr2)
    ax1.legend(loc='upper left')
    twin1.legend(loc='upper right')
    ax1.set_title(f"Highest Correlation: {highest_corr2} (Overall correlation = {df2[highest_corr2]:.2f})")

    # Plot lowest correlating Q signature
    ax2.plot(years, Qsig_obs['MRD'], label='Median Rank Distance')
    twin2 = ax2.twinx()
    twin2.plot(years, Qsig_obs[lowest_corr2].values, label=lowest_corr2, color='orange')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('MRD')
    twin2.set_ylabel(lowest_corr2)
    ax2.legend(loc='upper left')
    twin2.legend(loc='upper right')
    ax2.set_title(f"Lowest Correlation: {lowest_corr2} (Overall correlation = {df2[lowest_corr2]:.2f})")

    #scatterplot between MRD and Highest corr2
    plt.figure()
    sns.regplot(data = Qsig_obs, x = 'MRD', y = highest_corr2)
    #add text for R2 and p values
    slope,intercept,r,p,std = linregress(Qsig_obs['MRD'],Qsig_obs[highest_corr2])
    plt.text(0.1,0.9,f"r = {r:.2f}, p = {p:.2f}", transform = plt.gca().transAxes)
    plt.title(f"Correlation between {highest_corr2} and MRD")
    plt.grid()
    plt.show()
    

# Example usage
plot_correlations(LOA_objects, exp_id = self.ORIG_ID,
                    swe_metric = 'melt_sum_grid_MAPE', q_metric = 'NSE_meltseason')
#%%


Qsig_obs = self.Qsig_obs

# -- Now loop through each SWE metric, load MRD, compute correlations --
results = {}

for swe_metric in self.swe_metrics_to_use:
    # load posterior PRI for this SWE metric
    MD = SWE_PRI_dic[swe_metric]['PRI_SWE']
    MD = (
        MD[MD['category']=='Posterior']
        .drop(columns='category')
        .set_index('year')['PRI']
    )
    # align on years
    MD = MD.loc[years]
    # compute Pearson correlations between MRD and each Qsig_obs column
    corr_series = Qsig_obs.corrwith(MD)
    results[swe_metric] = corr_series

    # print sorted correlations
    print(f"\nCorrelations between MRD of {swe_metric} and Q/Climate signatures:")
    print(corr_series.sort_values(ascending=False))

# 'results' now holds a pd.Series of correlations for each swe_metric
heat_df = pd.DataFrame(results).T

# 2) Plot
plt.figure(figsize=(12, max(4, 0.5 * heat_df.shape[0])))
sns.heatmap(
    heat_df,
    annot=False,        # show the correlation values
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Pearson r"}
)
plt.xlabel("Q + Climate Signature")
plt.ylabel("SWE Metric")
plt.title("Correlation between MRD and Q/Climate Signatures")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%%
# self = LOA_objects['Syn_2911']
QSWE_corr = self.QSWE_corr
#filter QSWE_corr based on a list of SWE metric names
swe_metrics = ['melt_sum_elev_ME']
mask = QSWE_corr['level_0'].isin(swe_metrics)
QSWE_corr = QSWE_corr[mask]

plt.figure(figsize = (8,4))
# sns.stripplot(data = QSWE_corr, x = 'Year', 
#               y = 'correlation', hue = 'level_1',
#               palette = 'colorblind')
Q_ms = ['Qmean_meltseason_ME',
        'Qcv_meltseason_ME',
        'Qamp_ME',
        'KGE_meltseason']
for Qm in Q_ms:
    mask = QSWE_corr['level_1'] == Qm
    df = QSWE_corr[mask]
    plt.plot(df['Year'],df['correlation'], label = Qm)
plt.legend()
plt.title(f"Correlation between {swe_metrics[0]} and Q metrics")

#repeat the above, but for each combo in self.topcombos_tuples
QSWE_corr = self.QSWE_corr
#filter QSWE_corr based on a list of SWE metric names
plt.figure()
for combo in self.topcombos_tuples:
    mask= (QSWE_corr['level_0'] == combo[0]) & (QSWE_corr['level_1'] == combo[1])
    filtered_corr = QSWE_corr[mask]
    plt.plot(filtered_corr['Year'], filtered_corr['correlation'], label=f"{combo[0]} vs {combo[1]}")
plt.legend()



#%%
#reduce QSWE_corr to have Years on index, and level_1 on columns
QSWE_corr_pivot = QSWE_corr.pivot(index = 'Year', columns = 'level_1', 
                                  values = 'correlation')
topcombos = self.topcombos_tuples
QSWE_corr_pivot = QSWE_corr_pivot[topcombos]

corrtest = QSWE_corr_pivot.corr()
plt.figure(figsize = (8,8))
sns.heatmap(corrtest, annot = False, cmap = 'coolwarm')
plt.title("Correlation between the correlation of different Q metrics with melt_sum_ME")




#%%
#filter QSWE_corr based on a list of Q metric names
q_metrics = ['KGE_meltseason','NSE','t_hfd_ME',
                'Qmean_meltseason_ME',
                'bfi_ME']#'Qmean_ME','t_Qinflection_ME',,'Qcv_ME''Qcv_meltseason_ME',
q_combined = [
        'Qmean_meltseason_ME+\nbfi_ME',
                'Qmean_meltseason_ME+\nbfi_ME+\nt_hfd_ME',
            ]


# q_metrics = ['KGE_meltseason',
#                 'Qsum_meltseason_ME','Qcv_meltseason_ME',
#                 'Baseflow_index_ME','Half_flow_date_ME']

swe_metrics = [
            # 'SWE_max_elev_ME',
                'melt_sum_elev_ME',
            'SWE_7daymelt_elev_ME',
            't_SWE_max_elev_ME',
            #    'SWE_melt_KGE_elev',
            #    'SWE_NSE_elev',
                ]

# swe_metrics = [
#                 'melt_sum_ME',
#                'SWE_meltrate_ME',
#                't_SWE_max_ME']
swe_combined = [
    'melt_sum_elev_ME+\nSWE_7daymelt_elev_ME+\nt_SWE_max_elev_ME',
]
plotlength = 0.5*len(q_metrics+q_combined+swe_metrics+swe_combined)
pct = 0.01
m_rank_dic = {}
for year in range(self.START_YEAR, self.END_YEAR+1):
    m = self.metrics[year]
    m_rank = m.rank(ascending = True)
    for col in m_rank.columns:
        # if col in ['KGE','NSE','R2']:
        if 'KGE' in col or 'NSE' in col or 'R2' in col:
            m_rank[col] = m_rank[col].rank(ascending = False)
    for cc in q_combined+swe_combined:
        metrics = cc.split('+\n')
        rankie = sum(m_rank[m] for m in metrics).rank(ascending=True)
        m_rank[cc] = rankie
    m_rank_dic[year] = m_rank

rankstack = pd.DataFrame(columns=['QEM', 'SWEEM', 'Year', 'index', 'rank'])
rows = []

for qem in q_metrics+q_combined:
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        Nranks = int(np.ceil(pct * len(m_rank_dic[year])))
        indices = m_rank_dic[year][qem].nsmallest(Nranks).index
        for sweem in swe_metrics+swe_combined:
            for idx in indices:
                rank = m_rank_dic[year].loc[idx, sweem]
                rows.append({
                    'QEM': qem,
                    'SWEEM': sweem,
                    'Year': year,
                    'index': idx,
                    'rank': rank
                })
# Concatenate all rows into the rankstack DataFrame
rankstack = pd.concat([rankstack, pd.DataFrame(rows)], ignore_index=True)
#%%
rankstack_swe = pd.DataFrame(columns=['QEM', 'SWEEM', 'Year', 'index', 'rank'])
rows = []

for sweem in swe_metrics+swe_combined:
    for year in range(self.START_YEAR, self.END_YEAR + 1):
        Nranks = int(np.ceil(pct * len(m_rank_dic[year])))
        indices = m_rank_dic[year][sweem].nsmallest(Nranks).index
        for qem in q_metrics+q_combined:
            for idx in indices:
                rank = m_rank_dic[year].loc[idx, qem]
                rows.append({
                    'QEM': qem,
                    'SWEEM': sweem,
                    'Year': year,
                    'index': idx,
                    'rank': rank
                })
# Concatenate all rows into the rankstack DataFrame
rankstack_swe = pd.concat([rankstack_swe, pd.DataFrame(rows)], ignore_index=True)


print(rankstack)
#%%
f1,ax1 = plt.subplots(figsize = (5,plotlength))
sns.boxenplot(rankstack, hue = "SWEEM", y = 'QEM', x = 'rank', orient = 'h',
            palette = 'colorblind'  )
ax1.set_xlabel('Ranks')
ax1.set_ylabel('Q metric')
ax1.grid(alpha = 0.5)
# ax1.axvline(0, color = 'black',  linestyle = '--')
# ax1.set_xlim(-1,1)
ax1.legend(loc = (1.05,0.5))
ax1.set_title(f"SWE metrics rank for the {pct*100}% best Q metrics")
#%%
f1,ax1 = plt.subplots(figsize = (5,plotlength))
sns.boxenplot(rankstack_swe, hue = "QEM", y = 'SWEEM', x = 'rank', orient = 'h',
            palette = 'colorblind' )
ax1.set_xlabel('Ranks')
ax1.set_ylabel('Q metric')
ax1.grid(alpha = 0.5)
# ax1.axvline(0, color = 'black',  linestyle = '--')
# ax1.set_xlim(-1,1)
ax1.legend(loc = (1.05,0.5))
ax1.set_title(f"Q metrics rank for the {pct*100}% best SWE metrics")

#%% 
rankstack['QEM'] = pd.Categorical(rankstack['QEM'], categories=rankstack['QEM'].unique(), ordered=True)
rankstack['SWEEM'] = pd.Categorical(rankstack['SWEEM'], categories=rankstack['SWEEM'].unique(), ordered=True)
rankstack_swe['QEM'] = pd.Categorical(rankstack_swe['QEM'], categories=rankstack_swe['QEM'].unique(), ordered=True)
rankstack_swe['SWEEM'] = pd.Categorical(rankstack_swe['SWEEM'], categories=rankstack_swe['SWEEM'].unique(), ordered=True)
#take the mean rank for each year within rankstack 
f1,ax1 = plt.subplots(figsize = (5,plotlength))
meanrank = rankstack.groupby(['QEM','SWEEM','Year'])['rank'].mean().reset_index()
sns.stripplot(meanrank, hue = "SWEEM", y = 'QEM', x = 'rank', orient = 'h',
            dodge = True, alpha = 0.8,palette = 'colorblind')
sns.boxenplot(meanrank, hue = "SWEEM", y = 'QEM', x = 'rank', orient = 'h',
        fill = False, palette = 'colorblind', legend = False) 
ax1.set_xlabel('Ranks')
ax1.set_ylabel('Q metric')
ax1.grid(alpha = 0.5)
# ax1.axvline(0, color = 'black',  linestyle = '--')
# ax1.set_xlim(-1,1)
ax1.legend(loc = (1.05,0.5))
ax1.set_title(f"Mean yearly SWE metric rank for the {pct*100}% best Q metrics \n (each point is the mean rank for one year (2001-2022))")

#%% Same but inverse 
f1,ax1 = plt.subplots(figsize = (5,plotlength))
meanrank = rankstack_swe.groupby(['QEM','SWEEM','Year'])['rank'].mean().reset_index()
sns.stripplot(meanrank, hue = "QEM", y = 'SWEEM', x = 'rank', orient = 'h',
            dodge = True, alpha = 0.8,palette = 'colorblind')
sns.boxenplot(meanrank, hue = "QEM", y = 'SWEEM', x = 'rank', orient = 'h',
        fill = False, palette = 'colorblind', legend = False)
ax1.set_xlabel('Ranks')
ax1.set_ylabel('SWE metric')
ax1.grid(alpha = 0.5)
# ax1.axvline(0, color = 'black',  linestyle = '--')  
# ax1.set_xlim(-1,1)
ax1.legend(loc = (1.05,0.5))
ax1.set_title(f"Mean yearly Q metric rank for the {pct*100}% best SWE metrics \n (each point is the mean rank for one year (2001-2022))")

#%%
swetocheck = 'melt_sum_elev_ME+\nSWE_7daymelt_elev_ME+\nt_SWE_max_elev_ME'
# swetocheck = 'melt_sum_elev_ME'
fbig,axbig = plt.subplots(1,1,figsize = (7,12))
palette = sns.color_palette("viridis", n_colors = self.END_YEAR-self.START_YEAR+1)
l = []
for year in range(self.START_YEAR, self.END_YEAR+1):
    swe3_rank = m_rank_dic[year][swetocheck]
    swe3_idxs = swe3_rank.nsmallest(10).index
    pars = self.pars[year]
    norm_pars = (pars - pars.mean()) / pars.std()

    # f1,ax1 = plt.subplots()
    # sns.stripplot(data = norm_pars, color = 'grey',
    #                alpha = 0.8, orient = 'h', ax = ax1, size = 1,
    #                dodge = 10)
    norm_par_selection = norm_pars.loc[swe3_idxs]
    melted = norm_par_selection.melt()
    melted['Year'] = year
    l.append(melted)
    # sns.stripplot(data = norm_par_selection, color = 'tab:red',
    #                    orient = 'h', ax = ax1, size = 5,
    #                   dodge = 10)
    # sns.stripplot(data = norm_par_selection, color = palette[year-self.START_YEAR],
    #                alpha = 0.5, orient = 'h', ax = axbig, size = 15,
    #                marker = 'd',facecolor =  palette[year-self.START_YEAR],
    #                dodge = 10)
stacked_pars = pd.concat(l,axis=0)
sns.stripplot(data = stacked_pars, x = 'value', y = 'variable', hue = 'Year',
                    alpha = 0.5, orient = 'h', ax = axbig, size = 15,
                    marker = 'd',dodge = True,jitter = 0.1, palette = palette)
axbig.set_xlabel('Normalized parameter value')
axbig.set_ylabel('Run index')
axbig.grid()

#with Line2D, make one line of each color in palette and add the year as label
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=palette[i], lw=2) for i in range(len(palette))]
axbig.legend(custom_lines, 
                [str(year) for year in range(self.START_YEAR, self.END_YEAR+1)], 
                title = 'Year',
                loc = (1.05,0.2))
axbig.set_title(f"Parameter values for the 10 best runs for each year of \n {swetocheck}")


#%%
QEMs = ['Qmean_meltseason_ME','t_hfd_ME','bfi_ME']
SWEEMs = ['melt_sum_elev_ME','SWE_7daymelt_elev_ME','t_SWE_max_elev_ME']
corrdf = pd.DataFrame()
for em in QEMs+SWEEMs:
    # for sweem in SWEEMs:
    l = []
    for year in range(self.START_YEAR, self.END_YEAR+1):
        # idx = f"{em}_{year}"
        ranks = m_rank_dic[year][em].copy(deep = True)
        ranks.index = [f"{year}_{idx}" for idx in ranks.index]
        l.append(ranks)
    ranks = pd.concat(l,axis=0)
    corrdf[em] = ranks

#add the obs Q signatures to corrdf 
for year in range(self.START_YEAR, self.END_YEAR+1):
    qsig = self.Q_signatures[year]['obs']
    for idx in corrdf.index:
        YY = int(idx.split('_')[0])
        if year == YY:
            for col in qsig.index:
                corrdf.loc[idx,col] = qsig[col]
corrmatrix = corrdf.corr()
#%%
plt.figure(figsize = (10,10))
sns.heatmap(corrmatrix, cmap = 'coolwarm_r')

#%%
# We're gonna plot Q and SWE metrics against each other to see which ones we can leave out
# Or we can do PCA 
metrics = self.metrics[2019]
# Qmetrics = self.Qmetrics[2019]
# swemetrics = self.swemetrics[2019]

#load Qmetrics for all years and put in one df 


# \n {self.EXP_ID_translation[self.ORIG_ID]}')
# plt.title('Correlation between SWE metrics over all years \n (Synthetic case study 2001-2022 in Dischma)')
#%% Also we're gonna plot just the signatures against each other, without the metrics 
# Qsigs = self.Q_signatures[2019]['sim']
# swesigs = self.swe_signatures[2019]['sim']

Qsigs =pd.concat([self.Q_signatures[year]['sim'] for year in range(self.START_YEAR, self.END_YEAR+1)],axis=0)
q_metrics = ['Qmean', 'Qmax','t_hfd',
                    't_Qinflection','Qmean_meltseason','Qcv_meltseason',
                    'BaseFlow_sum',
                    'bfi','Qcv','Qamp','t_Qstart','t_Qrise']
Qsigs = Qsigs[q_metrics]
Qsigcorr = Qsigs.corr().abs()
# swesigcorr = swesigs.corr()

sns.heatmap(Qsigcorr, cmap = 'coolwarm')
# sns.heatmap(swesigcorr, cmap = 'coolwarm')

#%% plot bands in dem
import matplotlib.colors as colors

bands = self.elev_bands 
bands = np.arange(1700,3000,200)
bands = pd.Series(bands)
dem = self.E.dem
for i in range(len(bands)-1):
    a = bands[i]
    b = bands[i+1]
    dem = xr.where((dem >= a) & (dem < b), i, dem)

# Create a discrete colormap
cmap = plt.cm.get_cmap('turbo', len(bands)-1)  # Number of discrete colors equals number of bands
norm = colors.BoundaryNorm(np.arange(len(bands)), cmap.N)

# Create the plot with the discrete colormap
fig, ax = plt.subplots(figsize=(10, 8))
im = dem.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

# Add a custom colorbar with band elevation labels
cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(bands)-1) + 0.5)
cbar.set_ticklabels([f"{bands[i]:.0f} - {bands[i+1]:.0f} m" for i in range(len(bands)-1)])
cbar.set_label('Elevation bands')

plt.title('Elevation Bands')
plt.tight_layout()
plt.show()

#%%










#             ranks = rankstack[(rankstack['QEM'] == qem) & (rankstack['SWEEM'] == sweem)]
#             ranks.index = ranks['index']
#             corrdf[name] = ranks['rank']


# sns.stripplot(selection, x = 'rank', y = QEMaux, alpha = 0.5,orient = 'h')
# sns.scatterplot(selection, x = 'rank', y = QEMaux,  alpha = 0.5)




#%%
# P = self.meteo_base['pr'].mean(dim = ['lat','lon']).to_pandas()
# Q = m3s_to_mm(self.Qobs,self.E.dem_area).squeeze()#.loc[slice('2018-10-01','2019-09-30')]
# P = P.loc[Q.index]
# deficit = P.cumsum() - Q.cumsum()
# deficit_smoothed = deficit.rolling(window = 15).mean()

# diff = deficit.diff()
# diff_smoothed = diff.rolling(window = 15).min()


# test = deficit.diff()[deficit.diff()<0]




#%%



# dotty_plot('SWE_max_elev_ME','KGE')
# dotty_plot('SWE_max_elev_ME','Qmean_meltseason_ME')
# # dotty_plot('SWE_max_elev_ME','PBIAS')
# dotty_plot('melt_sum_elev_ME','Qmean_ME')
# dotty_plot('t_SWE_end_elev_ME','t_hfd_ME')
# dotty_plot('SWE_meltrate_elev_ME','t_hfd_ME')





















    #make a stripplot with one point per year for the correlation between each SWE metric with the Q KGE metric
    # #start by making one dataframe per year with the KGE and the SWE metrics
    # df_list = []
    # for year in range(self.START_YEAR, self.END_YEAR+1):
    #     corr_mat = self.metrics[year].corr()#.stack().reset_index(name='correlation')
    #     #keep only index with swe in it
    #     corr_mat = corr_mat.loc[corr_mat.index.str.contains('swe')]
    #     #keep only columns with KGE in it
    #     # corr_mat = corr_mat.loc[:,'KGE']
    #     df_list.append(corr_mat)
    # dfs = pd.concat(df_list,axis=1)
    # dfs.columns = range(self.START_YEAR, self.END_YEAR+1)
    # df = dfs.transpose()

    # f1,ax1 = plt.subplots()
    # sns.stripplot(data = df, jitter = 0.2, ax = ax1, orient='h')
    # ax1.set_xlabel('Pearson correlation')
    # ax1.set_ylabel('Year')
    # ax1.grid()
    # ax1.set_title('R2 between SWE metrics and Q KGE')
    # for label in ax1.get_xticklabels():
    #     label.set_rotation(45)

    #%%

    
    #%%





#     def Q1SWEmany_plot(Qmetric):




# dotty_plot('SWE_max_over_z_ME','KGE')
# dotty_plot('SWE_max_over_z_ME','PBIAS')
# dotty_plot('melt_sum_over_z_ME','Qmean_ME')
# dotty_plot('t_SWE_end_over_z_ME','t_hfd_ME')



# dotty_plot('Qmax_ME','SWE_max_ME')
# dotty_plot(')


#         df = self.metrics[self.START_YEAR][[metric1,metric2]]
#         f1, ax = plt.subplots()
#         sns.kdeplot(data = df, 
#                     x = metric1, 
#                     y = metric2, 
#                     ax = ax, 
#                     color = 'black',
#                     fill = True)




# for year in range(self.START_YEAR, self.END_YEAR+1):
#     self.metrics[year] = self.metrics[year].rename(columns = {
#         'bfi_ME':'Baseflow_index_ME',
#         't_hfd_ME':'Half_flow_date_ME'})



    # 'SWE_melt_KGE':'SWE melt KGE catchment',
    #  'SWE_melt_KGE_over_z':'SWE melt KGE',
    #  'SWE_NSE_over_z'})

# for year in range(self.START_YEAR, self.END_YEAR + 1):
#     self.metrics[year] = self.metrics[year].rename(columns=lambda x: x.replace('_elev', ''))



# f1,axswe = plt.subplots()
# swetime = slice(f"{year-1}-10-01", f"{year}-06-30")
# sweobs = self.SWEobs.mean(dim=['lat', 'lon']).sel(time=swetime).to_pandas()
# sweobs.plot(ax=axswe, color='black', zorder=1e6, legend=False)
# sweall = self.SWE[year].sel(time=swetime).mean(dim=['lat', 'lon']).to_pandas().drop(columns='spatial_ref')
# maxes = sweall.max(axis = 1)
# minusses = sweall.min(axis = 1)
# axswe.fill_between(sweall.index, minusses, maxes, color = 'grey', alpha = 0.2)
# # sweall.iloc[:,:int(0.5*len(sweobs))].plot(ax=axswe, color='grey', alpha=0.05, legend=False)
# #select indices from sweall, but only if they exist 
# adjusted_indices = [i for i in swe3_idxs if i in sweall.columns]
# swebest = sweall.loc[:, adjusted_indices]
# swebest.plot(ax=axswe, color='tab:blue', alpha=0.3, legend=False)













        # alpha = np.arange(0.9, 1, 0.01)
        # qb = pd.DataFrame({a: hs.baseflow(q_obs_mmd, alpha=a) for a in alpha})
        # qb.plot()

        # mrc_np, bfr_k_np = hs.baseflow_recession(q_obs_mmd, fit_method="nonparametric_analytic")
        # mrc_exp, bfr_k_exp = hs.baseflow_recession(q_obs_mmd, fit_method="exponential")

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

        # ax1.plot(mrc_np[:, 0], mrc_np[:, 1], "bx")
        # ax1.plot(
        #     np.sort(mrc_np[:, 0]),
        #     np.exp(np.log(mrc_np[0, 1]) - bfr_k_np * np.sort(mrc_np[:, 0])),
        #     "g-",
        #     linewidth=2,
        # )
        # ax1.set_xlabel("Relative time [days]")
        # ax1.set_ylabel("Flow [cfs]")
        # # ax1.set_xlim(0, 80)
        # # ax1.set_ylim(0, 80)
        # ax1.set_title("Non-parametric fitted recession curve")
        # ax1.legend(["MRC", f"Exponential fit (K={bfr_k_np:.4f})"])

        # ax2.plot(mrc_exp[:, 0], mrc_exp[:, 1], "bx")
        # ax2.plot(
        #     np.sort(mrc_exp[:, 0]),
        #     np.exp(np.log(mrc_exp[0, 1]) - bfr_k_exp * np.sort(mrc_exp[:, 0])),
        #     "g-",
        #     linewidth=2,
        # )
        # ax2.set_xlabel("Relative time [days]")
        # ax2.set_ylabel("Flow [cfs]")
        # # ax2.set_xlim(0, 80)
        # # ax2.set_ylim(0, 80)
        # ax2.set_title("Exponential fitted recession curve")
        # ax2.legend(["MRC", f"Exponential fit (K={bfr_k_exp:.4f})"])

        # fig.savefig(Path("_static", "recession.png"), bbox_inches="tight")

        # sig_obs.diff(sig_sim)

    # def load_meteo(self):



    # def load_SWE():

    

        
    

# Qsim= np.array(self.Q['ms0_0'][-365:])
# Qobs = np.array(self.Qobs[-365:])
# Qobs_smooth = np.roll(Qobs,5).mean()

# # f1,ax1 = plt.subplots()
# # Qsim.plot(ax = ax1)
# # Qobs.rolling(5).mean().plot(ax = ax1)
# # Qobs.plot(ax = ax1)

# freq = 1.0
# recession_length =3
# n_start = 0
# eps = 0.00
# start_of_recession = 'baseflow'
# lyne_hollick_smoothing = 0.925


# recsegsim = recseg(np.array(Qsim),
#                    freq = freq,
#                    recession_length = recession_length,
#                    n_start = n_start,
#                    eps = eps,
#                    start_of_recession = start_of_recession,
#                    lyne_hollick_smoothing = lyne_hollick_smoothing)
# print(recsegsim)


# plt.figure()
# plt.plot(Qsim)
# #plot red vertical lies for each first element of recsegsim  
# # and a blue one for each second element of recsegsim
# for seg in recsegsim:
#     plt.axvline(seg[0],color = 'red')
#     plt.axvline(seg[1],color = 'blue')
# plt.axvline(123,color = 'black')


# #%%
# from hydrosignatures.baseflow import _pad_array, __batch_forward
# streamflow = Qsim 
# len_decrease = recession_length / freq
# decreasing_flow = streamflow[1:] < (streamflow[:-1] + eps)
# start_point = np.where(~decreasing_flow)[0][0]
# decreasing_flow = decreasing_flow[start_point:]
# decreasing_flow[-1] = False

# flow_change = np.where(decreasing_flow[:-1] != decreasing_flow[1:])[0]
# flow_change = flow_change[: 2 * (len(flow_change) // 2)].reshape(-1, 2)

# flow_section = flow_change[(flow_change[:, 1] - flow_change[:, 0]) >= len_decrease + n_start]
# flow_section += start_point
# flow_section[:, 0] += n_start

# if start_of_recession == "baseflow":
#     pad_width = 10
#     q_bf = _pad_array(streamflow, pad_width)
#     q_bf = __batch_forward(np.atleast_2d(q_bf), lyne_hollick_smoothing)
#     q_bf = np.ascontiguousarray(q_bf[0, pad_width:-pad_width])
#     is_baseflow = np.isclose(q_bf, streamflow)
#     for i, (start, end) in enumerate(flow_section):
#         is_b_section = is_baseflow[start : end + 1]
#         if not np.any(is_b_section):
#             flow_section[i] = -1
#         else:
#             flow_section[i, 0] += np.argmax(is_b_section)

#     flow_section = flow_section[flow_section[:, 0] >= 0]
#     flow_section = flow_section[(flow_section[:, 1] - flow_section[:, 0]) >= 3]


# plt.figure()
# # # plt.plot(q_bf)
# # qq = streamflow[1:].copy()
# # qq[decreasing_flow] = np.nan
# # plt.plot(streamflow[1:], color = 'black')
# # plt.plot(qq, color = 'red')
# plt.plot(streamflow[1:])
# for seg in flow_section:
#     plt.axvline(seg[0],color = 'red')
#     plt.axvline(seg[1],color = 'blue')

#%%#test spatial metric 
#North-South distribution? 


def calc_Ipot_corr(self, swe_3d):
    # Calculate the mean SWE over time
    SWE2D = swe_3d.mean(dim='time')
    
    # Load and process Ipotmean
    Ipotmean = xr.open_dataset(join(self.CONTAINER, f"Ipot_DOY_{self.RESOLUTION}_{self.BASIN}.nc")).mean(dim='day_of_year')
    Ipotmean = Ipotmean.rename({'x': 'lon', 'y': 'lat'})['radiation']
    
    # Flatten the arrays and filter out NaNs
    swe2d = SWE2D.sortby(['lon', 'lat']).values.flatten()
    ipot = Ipotmean.sortby(['lon', 'lat']).values.flatten()
    mask = ~np.isnan(swe2d) & ~np.isnan(ipot)
    
    # Perform linear regression on the filtered data
    slope, intercept, r_value, p_value, std_err = linregress(swe2d[mask], ipot[mask])
  
    return slope


#correlate SWE2D to Ipotmean 
swe2d = SWE2D.to_pandas().stack()
ipot = Ipotmean.to_pandas().stack()
ipot = ipot.sort_index()
ipot.sort_values('lon')
swe2d = swe2d.loc[ipot.index]
correlation = swe2d.corr(ipot)
print(correlation)



plt.plot(swe2d_filtered, ipot_filtered, 'o')
plt.plot(swe2d_filtered, intercept + slope * swe2d_filtered, 'r')
plt.xlabel('SWE')
plt.ylabel('Ipotmean')
sns.scatterplot(x = swe2d, y = ipot, color = 'red')
sns.regplot(x=swe2d_filtered, y=ipot_filtered, scatter=False)
plt.show()

def filter_nan(s,o):
    data = np.transpose(np.array([s.flatten(),o.flatten()]))
    data = data[~np.isnan(data).any(1)]
    return data[:,0], data[:,1]
######################################################################################################################
def SPAEF(s, o):
    #remove NANs    
    s,o = filter_nan(s,o)
    
    bins=int(np.around(math.sqrt(len(o)),0))
    #compute corr coeff
    alpha = np.corrcoef(s,o)[0,1]
    #compute ratio of CV
    beta = variation(s)/variation(o)
    #compute zscore mean=0, std=1
    o=zscore(o)
    s=zscore(s)
    #compute histograms
    hobs,binobs = np.histogram(o,bins)
    hsim,binsim = np.histogram(s,bins)
    #convert int to float, critical conversion for the result
    hobs=np.float64(hobs)
    hsim=np.float64(hsim)
    #find the overlapping of two histogram      
    minima = np.minimum(hsim, hobs)
    #compute the fraction of intersection area to the observed histogram area, hist intersection/overlap index   
    gamma = np.sum(minima)/np.sum(hobs)
    #compute SPAEF finally with three vital components
    spaef = 1- np.sqrt( (alpha-1)**2 + (beta-1)**2 + (gamma-1)**2 )  

    return spaef, alpha, beta, gamma

import numpy as np
from scipy.stats import variation,zscore
import math
s = self.SWE[2001]['ms0_0'].mean(dim = 'time').values
o = self.SWEobs.mean(dim = 'time').values
s,o  = filter_nan(s,o)
spaef,alpha,beta,gamma = SPAEF(s,o)



def EOF_similarity(obs,model):
    array_size=obs.shape
# transform the datasets to its spatial anomalies which is an essential 
# step for the EOF analysis     
    for i in range(0,array_size[1]):
        obs[:,i]=obs[:,i]-np.mean(obs[:,i])
        model[:,i]=model[:,i]-np.mean(model[:,i])        
# Build the integral datamatrix that contains both datasets along the time axis    
    data=np.concatenate((obs,model),axis=1)
# Conduct the EOF via the svd function
    U, s, V = scipy.linalg.svd(data, full_matrices=True)
    Y = np.dot((data),np.transpose(V));
# Compute the amount of explained variance    
    var_exp=np.cumsum(s**2/np.sum(s**2));
# Split the resulting loadings into two    
    load_obs,load_model=np.array_split(np.transpose(V), 2)    
    dif=np.abs(load_obs-load_model)
# reverse cumsum of the explained variance    
    var=var_exp[0]
    for l in range(0,obs.shape[1]*2-1):
        var=np.append(var,(var_exp[l]-var_exp[l-1]))
# Compute the skill score based on the weighted sum of the absolute loading 
# differences and return skill (0 for perfect agreement). 
    skill=np.empty([1,1])   
    for l in range(0,obs.shape[1]):
        skill=np.append(skill,np.sum(dif[l,:]*var))
    skill=np.delete(skill,0,axis=0)        
# output: skill score, EOF maps 1-3, explained variance, loadings obs, loadings model
    return (skill,Y[:,0],Y[:,1],Y[:,2],var,load_obs,load_model) 

import scipy
s = self.SWE[2001]['ms0_0'].mean(dim='time').values
o = self.SWEobs.sel(time=slice('2000-10-01', '2001-09-30')).mean(dim='time').values

# Create a mask to filter out NaNs
mask = ~np.isnan(s) & ~np.isnan(o)

model = np.where(mask, s, 0)
obs = np.where(mask, o, 0)

skill, Y1, Y2, Y3, var, load_obs, load_model = EOF_similarity(obs, model)

# dem = self.E.dem
# from hydrobricks import Catchment as hbc
# CC = hbc()
# CC.extract_dem(dem_path)

# aspect = self.E.dem



#%%
Syn_Q = pd.read_csv(join(self.OUTDIR,'Synthetic_obs','Dischma_Q.csv'),
                    index_col = 0, parse_dates = True)
# %%
    # def calc_p_at_k(rank_stack,year,cat):
    #     rs = rank_stack[rank_stack['Year'] == year]
    #     selection_ranks = rs[rs['Category'] == cat]['Rank']
    #     k = selection_ranks.median()
    #     best_ranks = np.arange(1,k+1).astype(int)
    #     p_at_k = (selection_ranks <= k).sum()/k
    #     return p_at_k
    
    
    
    # p_at_k_values = {}
    # for year in range(self.START_YEAR, self.END_YEAR+1):
    #     p_at_k_value = calc_p_at_k(rank_stack_swe, year, 'Posterior')
    #     p_at_k_values[year] = p_at_k_value
    #     print(f"P@k for year {year}: {p_at_k_value}")



# Create the Figure for both Q PRI and SWE PRI 
plt.figure(figsize=(12, 10))

# Create stripplot with jittered points
ax = sns.stripplot(
    data=yearly_PRI_combined, 
    x='PRI', 
    y='SWE_metric',
    hue='PRI_type',
    palette=['tab:blue', 'tab:orange'],
    dodge=True,  # Separate SWE and Q points horizontally
    alpha=0.7,
    jitter=0.25,
    size=8
)

# Add mean values as markers with black edges
for pri_type, color in zip(['SWE PRI', 'Q PRI'], ['tab:blue', 'tab:orange']):
    means = yearly_PRI_combined[yearly_PRI_combined['PRI_type'] == pri_type].groupby('SWE_metric')['PRI'].mean()
    for metric in means.index:
        plt.scatter(
            means[metric], 
            metric, 
            marker='D',  # Diamond shape
            s=120,
            color=color,
            edgecolor='black',
            linewidth=1.5,
            zorder=10
        )

# Add vertical line at 0
plt.axvline(0, color='black', linestyle='--', alpha=0.7)

# Customize the plot
plt.grid(alpha=0.3)
plt.title('PRI Values by SWE Metric (Individual Years)', fontsize=14)
plt.xlabel('Posterior Rank Improvement (PRI)', fontsize=12)
plt.ylabel('SWE Metric', fontsize=12)

# Legend with custom labels
handles, labels = ax.get_legend_handles_labels()
custom_handles = handles + [
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:blue', 
               markeredgecolor='black', markersize=10),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:orange',
               markeredgecolor='black', markersize=10)
]
custom_labels = labels + ['SWE PRI Mean', 'Q PRI Mean']
plt.legend(custom_handles, custom_labels, title='PRI Type', loc='upper left', bbox_to_anchor=(1.05, 1))

# Adjust layout
plt.tight_layout()
plt.savefig('yearly_PRI_by_metric.png', dpi=300, bbox_inches='tight')
plt.show()





#%%