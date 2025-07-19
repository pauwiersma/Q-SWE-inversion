"""
Functions for calculating streamflow (Q) metrics and signatures.
These functions analyze various aspects of streamflow time series including:
- Seasonal characteristics (snowmelt season CV, sum)
- Statistical properties (skewness, variance, coefficient of variation)
- Timing metrics (peak timing, half flow date, inflection points)
- Flow distribution metrics (high flow frequency, peak distribution)
- Flow regime indicators (flashiness, baseflow, amplitude)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from hydrosignatures import flashiness_index as fi
from hydrosignatures.baseflow import baseflow

def calc_snowmeltseason_cv(q: pd.Series, months: list = [4, 5, 6, 7]) -> float:
    """
    Calculate the coefficient of variation (CV) for the snowmelt season.
    
    Args:
        q (pd.Series): Streamflow time series
        months (list): Months to include in snowmelt season (default: [4,5,6,7])
    
    Returns:
        float: Coefficient of variation for the snowmelt season
    """
    q_season = q.loc[q.index.month.isin(months)]
    cv = q_season.std() / q_season.mean()
    return cv

def calc_snowmeltseason_sum(q: pd.Series, months: list = [4, 5, 6, 7]) -> float:
    """
    Calculate the total streamflow during the snowmelt season.
    
    Args:
        q (pd.Series): Streamflow time series
        months (list): Months to include in snowmelt season (default: [4,5,6,7])
    
    Returns:
        float: Total streamflow during snowmelt season
    """
    q_season = q.loc[q.index.month.isin(months)]
    return q_season.sum()

def calc_q_skew(q: pd.Series) -> float:
    """
    Calculate the skewness of the streamflow time series.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Skewness coefficient
    """
    return q.skew()

def calc_q_var(q: pd.Series) -> float:
    """
    Calculate the variance of the streamflow time series.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Variance
    """
    return q.var()

def calc_q_cov(q: pd.Series) -> float:
    """
    Calculate the coefficient of variation (CV) of the streamflow time series.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Coefficient of variation (std/mean)
    """
    return q.std() / q.mean()

def calc_t_qmax(q: pd.Series) -> int:
    """
    Calculate the timing of maximum streamflow relative to October 1st.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Days since October 1st of the previous year
    """
    t_qmax = q.idxmax()
    return (t_qmax - pd.Timestamp(f"{t_qmax.year-1}-10-01")).days

def calc_half_flow_date(q: pd.Series) -> int:
    """
    Calculate the half flow date (HFD) - the date when half of the annual flow has passed.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Index of the half flow date, or np.nan if not found
    """
    q_half_sum = 0.5 * np.sum(q)
    q_cumsum = np.cumsum(q)
    hfd_aux = np.where(q_cumsum > q_half_sum)[0]
    return hfd_aux[0] if len(hfd_aux) > 0 else np.nan

def calc_half_flow_interval(q: pd.Series) -> int:
    """
    Calculate the time span between quarter and three-quarter flow dates.
    
    The interval is defined as the time between:
    - When cumulative discharge reaches 25% of annual total
    - When cumulative discharge reaches 75% of annual total
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Number of days between quarter and three-quarter flow dates
    """
    q_quarter = 0.25 * np.sum(q)
    q_three_quarters = 0.75 * np.sum(q)
    q_cumsum = np.cumsum(q)
    hfi_aux1 = np.where(q_cumsum > q_quarter)[0]
    hfi_aux2 = np.where(q_cumsum > q_three_quarters)[0]
    return hfi_aux2[0] - hfi_aux1[0]

def calc_high_flow_freq(q: pd.Series, pct: float = 0.9) -> float:
    """
    Calculate the frequency of high flow events.
    
    Args:
        q (pd.Series): Streamflow time series
        pct (float): Threshold percentile for high flows (default: 0.9)
    
    Returns:
        float: Frequency of high flow events (days exceeding threshold / total days)
    """
    q_threshold = pct * q.max()
    return np.sum(q > q_threshold) / len(q)

def calc_peak_distribution(q: pd.Series, slope_range: tuple = (0.1, 0.5), fit_log_space: bool = False) -> float:
    """
    Calculate the peak flow distribution metric.
    
    Args:
        q (pd.Series): Streamflow time series
        slope_range (tuple): Range for slope calculation (default: (0.1, 0.5))
        fit_log_space (bool): Whether to fit in log space (default: False)
    
    Returns:
        float: Peak distribution metric
    """
    peaks, _ = find_peaks(q)
    q_peak = q.iloc[peaks]
    q_peak_sorted = np.sort(q_peak)[-1::-1]
    q90 = np.quantile(q, 0.9)
    q50 = np.quantile(q, 0.5)
    return (q90 - q50) / 0.4

def calc_flashiness_index(q: pd.Series) -> float:
    """
    Calculate the flashiness index using the hydrosignatures package.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Flashiness index
    """
    return fi(q).item()

def calc_q_inflection(q: pd.Series) -> int:
    """
    Calculate the inflection point in the cumulative streamflow.
    
    Uses a 30-day smoothing window and finds the maximum of the second derivative
    of the cumulative flow, which is related to the timing of maximum SWE.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Index of the first inflection point
    """
    q_cumsum = np.cumsum(q).squeeze().rolling(window=30).mean()
    first_derivative = np.gradient(q_cumsum)
    second_derivative = np.gradient(first_derivative)
    return np.nanargmax(second_derivative)

def peakfilter_mask(array: np.ndarray) -> np.ndarray:
    """
    Create a mask for peak filtering of streamflow data.
    
    Args:
        array (np.ndarray): Input array
    
    Returns:
        np.ndarray: Boolean mask for peak filtering
    """
    diff = np.diff(array, prepend=0)
    posdif = (diff > 0) * diff
    mask1 = posdif > 2 * np.std(posdif)
    mask1_shifted = np.roll(mask1, -1)
    mask1_shifted2 = np.roll(mask1, -2)
    mask1_shifted3 = np.roll(mask1, -3)
    mask2 = mask1 | mask1_shifted | mask1_shifted2 | mask1_shifted3
    mask3 = array > np.quantile(array, 0.95)
    return mask2 | mask3

def peakfilter(q_array: pd.Series) -> pd.Series:
    """
    Filter peak flows from the streamflow time series.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        pd.Series: Filtered streamflow with peaks removed
    """
    q = q_array.squeeze().copy()
    mask = peakfilter_mask(q)
    filtered = np.where(mask, q, np.nan)
    q[mask] = np.nan
    return q

def baseflow_filter(q_array: pd.Series) -> pd.Series:
    """
    Extract baseflow from the streamflow time series using the hydrosignatures package.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        pd.Series: Baseflow component
    """
    q = q_array.squeeze().copy()
    bf = baseflow(q)
    if np.all(bf == q):
        print("Warning: Baseflow equals total flow. Try using q_array[:,0]")
    return bf

def calc_q_amplitude(q_array: pd.Series) -> float:
    """
    Calculate the amplitude of the streamflow time series.
    
    Uses a 30-day rolling mean to smooth the data before calculating amplitude.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        float: Flow amplitude (max - min of smoothed series)
    """
    q = q_array.squeeze().copy()
    q_smoothed = q.rolling(window=30).mean()
    return q_smoothed.max() - q_smoothed.min()

def calc_qstart(q_array: pd.Series) -> int:
    """
    Calculate the start of the streamflow season.
    
    Uses a 30-day smoothing window and finds the intersection with the 5th percentile
    of the smoothed flow.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        int: Index of flow start, or np.nan if not found
    """
    q = q_array.squeeze().copy()
    window = 30
    q_smoothed = q.rolling(window=window).mean()
    pct = np.nanpercentile(q_smoothed, 5)
    intercepts = np.where(np.diff(np.sign(q_smoothed - pct)))[0]
    intercepts = intercepts[intercepts > window]
    
    if len(intercepts) == 0:
        return np.nan
    elif len(intercepts) == 2:
        return intercepts[1]
    elif len(intercepts) == 1:
        print("Warning: Only one intercept with Q05")
        return intercepts[0] if q_smoothed[intercepts[0]] > q_smoothed[intercepts[0]-1] else np.nan
    else:  # len(intercepts) > 2
        distances = np.diff(intercepts)
        return intercepts[np.argmax(distances)]

def calc_t_qrise(q_array: pd.Series) -> int:
    """
    Calculate the time between flow start and maximum flow.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        int: Number of days between flow start and maximum
    """
    qstart = calc_qstart(q_array)
    q = q_array.squeeze().copy()
    q_smoothed = q.rolling(window=30).mean()
    t_qmax = np.nanargmax(q_smoothed)
    return t_qmax - qstart