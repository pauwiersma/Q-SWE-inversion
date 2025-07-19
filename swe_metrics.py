"""
Module containing functions for calculating SWE (Snow Water Equivalent) metrics and signatures.

This module provides a collection of functions for analyzing snow-related metrics including:
- SWE maximum values and timing
- Snow water storage calculations
- Melt rates and timing
- Performance metrics (KGE, NSE) for snow simulations
- Elevation-based analyses

The functions are designed to work with both time series (1D) and spatial (3D) data.
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, Optional
import pandas as pd
# from scipy.stats import pearsonr, variation

# --- SWE Metrics and Signatures ---

def calc_sum2d(swe3d, dem=None):
    """
    Calculate the sum of SWE over latitude and longitude, normalized by the number of valid cells.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        dem: DEM array for valid cell count (optional)
    Returns:
        Sum of SWE normalized by cell count
    """
    sum2d = swe3d.sum(dim=['lat', 'lon'])
    if dem is not None:
        cell_count = np.sum(~np.isnan(dem))
        sum2d = sum2d / cell_count
    return sum2d

def calc_t_swe_max(swe_t):
    """
    Calculate the time index of maximum SWE.
    Args:
        swe_t: SWE time series
    Returns:
        Index of maximum SWE or np.nan
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    t_swe_max = np.nanargmax(swe_t)
    return t_swe_max if not np.isnan(t_swe_max) else np.nan

def calc_t_swe_max_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the time of maximum SWE for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        Time of max SWE per elevation band
    """
    t_swe_max = xr.apply_ufunc(calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
    t_swe_max_vs_elevation = calc_var_vs_elevation(t_swe_max)
    return t_swe_max_vs_elevation

def calc_t_swe_max_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the time of maximum SWE for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        Time index of max SWE for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    t_swe_max = calc_t_swe_max(swe_t)
    return t_swe_max

def calc_t_swe_max_grid(swe3d):
    """
    Calculate the time of maximum SWE for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        Time index of max SWE for each grid cell
    """
    t_swe_max_grid = xr.apply_ufunc(calc_t_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
    return t_swe_max_grid

def calc_swe_max(swe_t):
    """
    Calculate the maximum SWE.
    Args:
        swe_t: SWE time series
    Returns:
        Maximum SWE (or np.nan if all nan)
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    SWE_max = np.nanmax(swe_t)
    return SWE_max

def calc_swe_max_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the maximum SWE for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        Maximum SWE for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    SWE_max = calc_swe_max(swe_t)
    return SWE_max

def calc_swe_max_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the maximum SWE for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        Maximum SWE per elevation band
    """
    SWE_max = xr.apply_ufunc(calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
    SWE_max_vs_elevation = calc_var_vs_elevation(SWE_max)
    return SWE_max_vs_elevation

def calc_swe_max_grid(swe3d):
    """
    Calculate the maximum SWE for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        Maximum SWE per grid cell
    """
    SWE_max_grid = xr.apply_ufunc(calc_swe_max, swe3d, input_core_dims=[['time']], vectorize=True)
    return SWE_max_grid

def calc_sws(swe_t):
    """
    Calculate Snow Water Storage (SWS), the area under the SWE curve.
    Args:
        swe_t: SWE time series
    Returns:
        SWS (or np.nan if all nan)
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    sws = np.nansum(swe_t)
    return sws

def calc_sws_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate Snow Water Storage (SWS) for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        SWS per elevation band
    """
    sws = xr.apply_ufunc(calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)
    sws_vs_elevation = calc_var_vs_elevation(sws)
    return sws_vs_elevation

def calc_sws_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the total Snow Water Storage (SWS) for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        SWS for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    sws = calc_sws(swe_t)
    return sws

def calc_sws_grid(swe3d):
    """
    Calculate Snow Water Storage (SWS) for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        SWS per grid cell
    """
    sws_grid = xr.apply_ufunc(calc_sws, swe3d, input_core_dims=[['time']], vectorize=True)
    return sws_grid

def calc_melt_rate(swe_t):
    """
    Calculate the mean melt rate of SWE (negative diff averaged).
    Args:
        swe_t: SWE time series
    Returns:
        Mean melt rate (or np.nan if all nan)
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    diff = np.diff(swe_t)
    neg_diff = diff[diff < 0]
    mean_melt_rate = np.mean(neg_diff) * (-1) if len(neg_diff) > 0 else np.nan
    return mean_melt_rate

def calc_melt_rate_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the mean melt rate of SWE for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        Mean melt rate per elevation band
    """
    melt_rate = xr.apply_ufunc(calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)
    melt_rate_vs_elevation = calc_var_vs_elevation(melt_rate)
    return melt_rate_vs_elevation

def calc_melt_rate_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the mean melt rate of SWE for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        Mean melt rate for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    melt_rate = calc_melt_rate(swe_t)
    return melt_rate

def calc_melt_rate_grid(swe3d):
    """
    Calculate the mean melt rate of SWE for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        Mean melt rate per grid cell
    """
    melt_rate_grid = xr.apply_ufunc(calc_melt_rate, swe3d, input_core_dims=[['time']], vectorize=True)
    return melt_rate_grid

def calc_7_day_melt(swe_t):
    """
    Calculate the maximum 7-day melt rate of SWE.
    Args:
        swe_t: SWE time series
    Returns:
        Maximum 7-day melt rate (or np.nan if all nan)
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    diff = np.diff(swe_t)
    neg_diff = diff[diff < 0] * (-1)
    rol7 = pd.Series(neg_diff).rolling(window=7, min_periods=1).max().max() if len(neg_diff) > 0 else np.nan
    return rol7

def calc_7_day_melt_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the maximum 7-day melt rate of SWE for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        Maximum 7-day melt rate per elevation band
    """
    melt7 = xr.apply_ufunc(calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)
    melt7_vs_elevation = calc_var_vs_elevation(melt7)
    return melt7_vs_elevation

def calc_7_day_melt_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the maximum 7-day melt rate of SWE for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        Maximum 7-day melt rate for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    melt7 = calc_7_day_melt(swe_t)
    return melt7

def calc_7_day_melt_grid(swe3d):
    """
    Calculate the maximum 7-day melt rate of SWE for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        Maximum 7-day melt rate per grid cell
    """
    melt7_grid = xr.apply_ufunc(calc_7_day_melt, swe3d, input_core_dims=[['time']], vectorize=True)
    return melt7_grid

def calc_melt_sum(swe_t):
    """
    Calculate the total melt sum (sum of negative diff).
    Args:
        swe_t: SWE time series
    Returns:
        Total melt sum
    """
    diff = np.diff(swe_t)
    pos_swe = diff[diff < 0]
    melt_sum = np.nansum(pos_swe) * (-1)
    return melt_sum

def calc_sf_sum(swe_t):
    """
    Calculate the total snowfall sum (sum of positive diff).
    Args:
        swe_t: SWE time series
    Returns:
        Total snowfall sum
    """
    diff = np.diff(swe_t)
    pos_swe = diff[diff > 0]
    sf_sum = np.nansum(pos_swe)
    return sf_sum

def calc_melt_sum_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the total melt sum for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        Total melt sum per elevation band
    """
    melt_sum = xr.apply_ufunc(calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
    melt_sum = xr.where(melt_sum > 0, melt_sum, np.nan)
    melt_sum_vs_elevation = calc_var_vs_elevation(melt_sum)
    return melt_sum_vs_elevation

def calc_melt_sum_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the total melt sum for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        Total melt sum for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    melt_sum = calc_melt_sum(swe_t)
    return melt_sum

def calc_melt_sum_grid(swe3d):
    """
    Calculate the total melt sum for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        Total melt sum per grid cell (nan if all swe is nan)
    """
    melt_sum_grid = xr.apply_ufunc(calc_melt_sum, swe3d, input_core_dims=[['time']], vectorize=True)
    melt_sum_grid = xr.where(swe3d.isnull().all(dim='time'), np.nan, melt_sum_grid)
    return melt_sum_grid

def calc_swe_dates(swe_t, smooth_window=5, threshold_frac=0.1):
    """
    Identify start and end of SWE season based on smoothed SWE values and thresholding.
    Parameters:
        swe_t: SWE time series (array-like)
        smooth_window (int): Window size for smoothing (default 5)
        threshold_frac (float): Fraction of max SWE to use as threshold (default 0.1)
    Returns:
        list: [start_index, end_index] of the main snow period, or [np.nan, np.nan] if not identifiable.
    """
    if swe_t is None or len(swe_t) < 3 or np.all(np.isnan(swe_t)):
        return [np.nan, np.nan]
    swe_t = np.array(swe_t)
    swe_max = np.nanmax(swe_t)
    if swe_max <= 0:
        return [np.nan, np.nan]
    swe_smooth = pd.Series(swe_t).rolling(window=smooth_window, center=True, min_periods=1).mean()
    threshold = threshold_frac * swe_max
    mask = swe_smooth > threshold
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
    longest = max(segments, key=lambda x: x[1] - x[0])
    return [longest[0], longest[1]]

def calc_swe_start(swe_t):
    """
    Calculate the start date of SWE (using calc_swe_dates).
    Args:
        swe_t: SWE time series
    Returns:
        Start index (or np.nan if all nan)
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    intercepts = calc_swe_dates(swe_t)
    return intercepts[0]

def calc_swe_end(swe_t):
    """
    Calculate the end date of SWE (using calc_swe_dates).
    Args:
        swe_t: SWE time series
    Returns:
        End index (or np.nan if all nan)
    """
    if np.all(np.isnan(swe_t)):
        return np.nan
    intercepts = calc_swe_dates(swe_t)
    return intercepts[1]

def calc_swe_start_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the start date of SWE for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        Start index per elevation band
    """
    starts = xr.apply_ufunc(calc_swe_start, swe3d, input_core_dims=[['time']], vectorize=True, output_dtypes=[np.float64])
    starts_vs_elevation = calc_var_vs_elevation(starts)
    return starts_vs_elevation

def calc_swe_end_vs_elevation(swe3d, calc_var_vs_elevation):
    """
    Calculate the end date of SWE for different elevation bands.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
    Returns:
        End index per elevation band
    """
    ends = xr.apply_ufunc(calc_swe_end, swe3d, input_core_dims=[['time']], vectorize=True, output_dtypes=[np.float64])
    ends_vs_elevation = calc_var_vs_elevation(ends)
    return ends_vs_elevation

def calc_swe_start_end_catchment(swe3d, calc_sum2d_func):
    """
    Calculate the start and end dates of SWE for the entire catchment.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
    Returns:
        list: [start_index, end_index] for catchment
    """
    swe_t = calc_sum2d_func(swe3d)
    intercepts = calc_swe_dates(swe_t)
    return intercepts

def calc_swe_start_grid(swe3d):
    """
    Calculate the start date of SWE for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        Start index per grid cell
    """
    starts_grid = xr.apply_ufunc(calc_swe_start, swe3d, input_core_dims=[['time']], vectorize=True)
    return starts_grid

def calc_swe_end_grid(swe3d):
    """
    Calculate the end date of SWE for each grid cell.
    Args:
        swe3d: 3D SWE data (time, lat, lon)
    Returns:
        End index per grid cell
    """
    ends_grid = xr.apply_ufunc(calc_swe_end, swe3d, input_core_dims=[['time']], vectorize=True)
    return ends_grid

# --- KGE and NSE (Melt) ---

def calc_melt_kge(swe_obs_t, swe_sim_t, he):
    """
    Calculate the Kling-Gupta Efficiency (KGE) for melt rates.
    Args:
        swe_obs_t: Observed SWE time series
        swe_sim_t: Simulated SWE time series
        he: hydroeval module (or equivalent) for KGE calculation
    Returns:
        KGE value (or np.nan if not computable)
    """
    dif_obs = np.diff(swe_obs_t)
    dif_sim = np.diff(swe_sim_t)
    neg_dif_obs = np.where(dif_obs < 0, dif_obs * (-1), 0)
    neg_dif_sim = np.where(dif_sim < 0, dif_sim * (-1), 0)
    kge = he.kge_2009(neg_dif_obs, neg_dif_sim)
    return kge

def calc_melt_kge_vs_elevation(swe_obs, swe_sim, calc_var_vs_elevation, he):
    """
    Calculate the KGE for melt rates for different elevation bands.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
        he: hydroeval module (or equivalent) for KGE calculation
    Returns:
        KGE per elevation band
    """
    kges = xr.apply_ufunc(calc_melt_kge, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64], kwargs={'he': he})
    kges_vs_elevation = calc_var_vs_elevation(kges)
    return kges_vs_elevation

def calc_melt_kge_catchment(swe_obs, swe_sim, calc_sum2d_func, he):
    """
    Calculate the KGE for melt rates for the entire catchment.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
        he: hydroeval module (or equivalent) for KGE calculation
    Returns:
        KGE for catchment
    """
    swe_obs_t = calc_sum2d_func(swe_obs).to_pandas()
    swe_sim_t = calc_sum2d_func(swe_sim).to_pandas()
    kge = calc_melt_kge(swe_obs_t, swe_sim_t, he)
    return kge

def calc_melt_kge_grid(swe_obs, swe_sim, he):
    """
    Calculate the KGE for melt rates for each grid cell.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        he: hydroeval module (or equivalent) for KGE calculation
    Returns:
        KGE per grid cell
    """
    kge_grid = xr.apply_ufunc(calc_melt_kge, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, kwargs={'he': he})
    return kge_grid

# --- NSE (Melt) ---

def calc_melt_nse(swe_obs_t, swe_sim_t, he):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) for melt rates.
    Args:
        swe_obs_t: Observed SWE time series
        swe_sim_t: Simulated SWE time series
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE value (or np.nan if not computable)
    """
    dif_obs = np.diff(swe_obs_t)
    dif_sim = np.diff(swe_sim_t)
    neg_dif_obs = np.where(dif_obs < 0, dif_obs * (-1), 0)
    neg_dif_sim = np.where(dif_sim < 0, dif_sim * (-1), 0)
    nse = he.nse(neg_dif_obs, neg_dif_sim)
    return nse

def calc_melt_nse_vs_elevation(swe_obs, swe_sim, calc_var_vs_elevation, he):
    """
    Calculate the NSE for melt rates for different elevation bands.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE per elevation band
    """
    nses = xr.apply_ufunc(calc_melt_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64], kwargs={'he': he})
    nses_vs_elevation = calc_var_vs_elevation(nses)
    return nses_vs_elevation

def calc_melt_nse_catchment(swe_obs, swe_sim, calc_sum2d_func, he):
    """
    Calculate the NSE for melt rates for the entire catchment.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE for catchment
    """
    swe_obs_t = calc_sum2d_func(swe_obs).to_pandas()
    swe_sim_t = calc_sum2d_func(swe_sim).to_pandas()
    nse = calc_melt_nse(swe_obs_t, swe_sim_t, he)
    return nse

def calc_melt_nse_grid(swe_obs, swe_sim, he):
    """
    Calculate the NSE for melt rates for each grid cell.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE per grid cell
    """
    nse_grid = xr.apply_ufunc(calc_melt_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, kwargs={'he': he})
    return nse_grid

# --- NSE (Snowfall) ---

def calc_snowfall_nse(swe_obs_t, swe_sim_t, he):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) for snowfall (positive diff).
    Args:
        swe_obs_t: Observed SWE time series
        swe_sim_t: Simulated SWE time series
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE value (or np.nan if not computable)
    """
    dif_obs = np.diff(swe_obs_t)
    dif_sim = np.diff(swe_sim_t)
    pos_dif_obs = dif_obs[dif_obs > 0]
    pos_dif_sim = dif_sim[dif_sim > 0]
    if len(pos_dif_obs) == 0 or len(pos_dif_sim) == 0:
        return np.nan
    nse = he.nse(pos_dif_obs, pos_dif_sim)
    return nse

def calc_snowfall_nse_vs_elevation(swe_obs, swe_sim, calc_var_vs_elevation, he):
    """
    Calculate the NSE for snowfall for different elevation bands.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE per elevation band
    """
    nses = xr.apply_ufunc(calc_snowfall_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64], kwargs={'he': he})
    nses_vs_elevation = calc_var_vs_elevation(nses)
    return nses_vs_elevation

def calc_snowfall_nse_catchment(swe_obs, swe_sim, calc_sum2d_func, he):
    """
    Calculate the NSE for snowfall for the entire catchment.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE for catchment
    """
    swe_obs_t = calc_sum2d_func(swe_obs).to_pandas()
    swe_sim_t = calc_sum2d_func(swe_sim).to_pandas()
    nse = calc_snowfall_nse(swe_obs_t, swe_sim_t, he)
    return nse

def calc_snowfall_nse_grid(swe_obs, swe_sim, he):
    """
    Calculate the NSE for snowfall for each grid cell.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE per grid cell
    """
    nse_grid = xr.apply_ufunc(calc_snowfall_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, kwargs={'he': he})
    return nse_grid

# --- NSE (SWE) ---

def calc_swe_nse(swe_obs_t, swe_sim_t, he):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) for SWE.
    Args:
        swe_obs_t: Observed SWE time series
        swe_sim_t: Simulated SWE time series
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE value (or np.nan if not computable)
    """
    nse = he.nse(swe_obs_t, swe_sim_t)
    return nse

def calc_swe_nse_vs_elevation(swe_obs, swe_sim, calc_var_vs_elevation, he):
    """
    Calculate the NSE for SWE for different elevation bands.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_var_vs_elevation: function to aggregate by elevation
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE per elevation band
    """
    nses = xr.apply_ufunc(calc_swe_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, output_dtypes=[np.float64], kwargs={'he': he})
    nses_vs_elevation = calc_var_vs_elevation(nses)
    return nses_vs_elevation

def calc_swe_nse_catchment(swe_obs, swe_sim, calc_sum2d_func, he):
    """
    Calculate the NSE for SWE for the entire catchment.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        calc_sum2d_func: function to sum over space
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE for catchment
    """
    swe_obs_t = calc_sum2d_func(swe_obs).to_pandas()
    swe_sim_t = calc_sum2d_func(swe_sim).to_pandas()
    nse = calc_swe_nse(swe_obs_t, swe_sim_t, he)
    return nse

def calc_swe_nse_grid(swe_obs, swe_sim, he):
    """
    Calculate the NSE for SWE for each grid cell.
    Args:
        swe_obs: Observed 3D SWE data (time, lat, lon)
        swe_sim: Simulated 3D SWE data (time, lat, lon)
        he: hydroeval module (or equivalent) for NSE calculation
    Returns:
        NSE per grid cell
    """
    nse_grid = xr.apply_ufunc(calc_swe_nse, swe_obs, swe_sim, input_core_dims=[['time'], ['time']], vectorize=True, kwargs={'he': he})
    return nse_grid

# --- Elevation and SPAEF ---

def calc_elev_bands(dem, N_bands=10):
    """
    Calculate elevation bands, making sure each band has the same number of pixels.
    
    Args:
        dem (xr.DataArray): Digital elevation model data
        N_bands (int): Number of elevation bands (default 10)
    
    Returns:
        np.ndarray: Array of elevation band boundaries
    """
    elev_flat = dem.values.flatten()
    elev_flat = pd.Series(elev_flat[~np.isnan(elev_flat)]).sort_values().reset_index(drop=True)
    quantiles = elev_flat.quantile(np.linspace(0, 1, N_bands + 1)).reset_index(drop=True)
    rounded_quantiles = np.round(quantiles, 0)
    return rounded_quantiles

def calc_var_vs_elevation(var2d, dem, elev_bands=None):
    """
    Calculate the mean of a variable in elevation bands defined by the calculated elevation bands.
    
    Args:
        var2d (xr.DataArray): 2D variable to analyze
        dem (xr.DataArray): Digital elevation model data
        elev_bands (np.ndarray, optional): Elevation band boundaries. If None, will be calculated using calc_elev_bands
    
    Returns:
        pd.DataFrame: Mean values of the variable for each elevation band
    """
    var2d_flat = var2d.values.flatten()
    dem_flat = dem.values.flatten()

    df = pd.DataFrame({'elevation': dem_flat, 'var': var2d_flat})
    df = df.dropna()
    
    if elev_bands is None:
        elev_bands = calc_elev_bands(dem)
    
    df['elevation_band'] = pd.cut(df['elevation'], bins=elev_bands, labels=False)

    var_by_band = (
        df.groupby('elevation_band')['var']
        .mean()
        .rename_axis('elevation_band')
        .reset_index(name='var')
        .set_index('elevation_band')
    )
    return var_by_band

def calculate_spaef(swe_obs, swe_sim, calc_melt_sum_grid):
    """
    Calculate the SPAEF metric for assessing spatial patterns of SWE.
    Takes 3D data, turns it into 2D with melt_sum, and then calculates the SPAEF.
    
    Args:
        swe_obs (xr.DataArray): Observed 3D SWE data
        swe_sim (xr.DataArray): Simulated 3D SWE data
        calc_melt_sum_grid (callable): Function to calculate melt sum grid
    
    Returns:
        float: SPAEF value
    """
    obs = calc_melt_sum_grid(swe_obs)
    sim = calc_melt_sum_grid(swe_sim)

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

    # Calculate SPAEF
    SPAEF = 1 - np.sqrt((A - 1)**2 + (B - 1)**2 + (C - 1)**2)
    return SPAEF

def calculate_kge(sim, obs, he):
    """
    Calculate the Kling-Gupta Efficiency (KGE) using the hydroeval module.
    
    Args:
        sim (np.ndarray): Simulated values
        obs (np.ndarray): Observed values
        he: hydroeval module for KGE calculation
    
    Returns:
        float: KGE value
    """
    return he.kge_2009(sim, obs)

def calculate_nse(sim, obs, he):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) using the hydroeval module.
    
    Args:
        sim (np.ndarray): Simulated values
        obs (np.ndarray): Observed values
        he: hydroeval module for NSE calculation
    
    Returns:
        float: NSE value
    """
    return he.nse(sim, obs) 