#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 06:46:09 2023

@author: pwiersma
"""

import logging
import warnings
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("esmvalcore")
logger.setLevel(logging.WARNING)
# import ewatercycle.forcing
# import ewatercycle.models
import matplotlib.pyplot as plt
from typing import Tuple, Union
from matplotlib.dates import DateFormatter
import seaborn as sns
import matplotlib.colors as mcolors



def cluster_hydrograph_ranges(
    discharge: pd.DataFrame,
    *,
    reference: str,
    clusters = int,
    precipitation: pd.DataFrame = None,
    sim_names = list,
    dpi: int = None,
    title: str = "Hydrograph",
    discharge_units: str = "m$^3$ s$^{-1}$",
    precipitation_units: str = "mm day$^{-1}$",
    figsize: Tuple[float, float] = (10, 10),
    filename: Union[os.PathLike, str] = None,
    grid : bool = True,
    legend_loc : Tuple[float,float] = (1.10,1),
    logy : bool = False,
    model_labels = None,
    **kwargs,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot a hydrograph.

    This utility function makes it convenient to create a hydrograph from
    a set of discharge data from a `pandas.DataFrame`. A column must be marked
    as the reference, so that the agreement metrics can be calculated.

    Optionally, the corresponding precipitation data can be plotted for
    comparison.

    Parameters
    ----------
    discharge : list
        Dataframe containing time series of discharge data to be plotted.
    reference : str
        Name of the reference data, must correspond to a column in the discharge
        dataframe. Metrics are calculated between the reference column and each
        of the other columns.
    precipitation : pd.DataFrame, optional
        Optional dataframe containing time series of precipitation data to be
        plotted from the top of the hydrograph.
    dpi : int, optional
        DPI for the plot.
    title : str, optional
        Title of the hydrograph.
    discharge_units : str, optional
        Units for the discharge data.
    precipitation_units : str, optional
        Units for the precipitation data.
    figsize : (float, float), optional
        With, height of the plot in inches.
    filename : str or Path, optional
        If specified, a copy of the plot will be saved to this path.
    **kwargs:
        Options to pass to the matplotlib plotting function

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    ax, ax_tbl : tuple of `matplotlib.axes.Axes`
    """

    cmap = mcolors.ListedColormap(sns.color_palette('Blues_d'))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap.colors)
    
    fig, ax = plt.subplots(
        dpi=dpi,
        figsize=figsize
    )

    ax.set_title(title,fontdict = {'fontsize':'xx-large'})
    ax.set_ylabel(f"Discharge ({discharge_units})")
    Global_label, Yearly_label = model_labels
    
    # for i,discharge in enumerate(discharges):

    discharge_cols = discharge.columns.drop(reference)
    y_obs = discharge[reference]
    y_sim = discharge[discharge_cols]
    # if i==0:        
    y_obs.plot(ax=ax,color = 'black',linestyle = 'dashed',label = 'Observed')#, **kwargs)
    
    cmap = mcolors.ListedColormap(sns.color_palette('Oranges_d'))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap.colors)
    for k in range(clusters):
        y_sim[f'Global_c{k}'].plot(ax = ax,label = (k==2)*Global_label, alpha = 1,
                                   linestyle = 'dotted',color = cmap(k))
    
    #(k==0) *f'Cluster centers {k}-{clusters} ' 
    cmap = mcolors.ListedColormap(sns.color_palette('Blues_d'))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap.colors)
    for k in range(clusters):
        # color = ax.get_children()[k+1].get_color()
        cols  = discharge_cols[discharge_cols.str.contains(f'R_c{k}')]
        y_sim_min = discharge[cols].min(axis = 1)
        y_sim_max = discharge[cols].max(axis = 1)
        ax.fill_between(y_sim_min.index,
                        y_sim_min,
                        y_sim_max,
                        alpha = 0.6, 
                        color = cmap(k),
                        label = (k==2)* Yearly_label)
    # (k==0) *f'Reconstruction ensembles {k}-{clusters} '
    if logy ==True:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0,ax.get_ylim()[1] * 0.8)

    
    
    
    # if 'Benchmark' in discharge_cols:
    #     y_sim['Benchmark'].plot(ax = ax, color = 'tab:red', label = 'Benchmark')
    #     y_sim.columns.drop('Benchmark')
    
    # y_sim_min = discharge[discharge_cols].min(axis = 1)
    # y_sim_max = discharge[discharge_cols].max(axis = 1)
    
    # # y_sim_min = discharge[discharge_cols].quantile(0.05,axis = 1)
    # # y_sim_max = discharge[discharge_cols].quantile(0.95,axis = 1)
    # y_sim_median = discharge[discharge_cols].median(axis = 1)
    # medianplot = ax.plot(y_sim_median.index,y_sim_median)

    
    # y_sim.plot(ax=ax, **kwargs)
    

    handles, labels = ax.get_legend_handles_labels()

    # Add precipitation as bar plot to the top if specified
    if precipitation is not None:
        ax_pr = ax.twinx()
        ax_pr.invert_yaxis()
        ax_pr.set_ylabel(f"Precipitation ({precipitation_units})")
        prop_cycler = ax._get_lines.prop_cycler

        for pr_label, pr_timeseries in precipitation.iteritems():
            #color = next(prop_cycler)["color"]
            color = 'tab:blue'
            ax_pr.bar(
                pr_timeseries.index.values,
                pr_timeseries.values,
                alpha=0.4,
                label=pr_label,
                color=color,
            )

        # tweak ylim to make space at bottom and top
        ax_pr.set_ylim(ax_pr.get_ylim()[0] * (7 / 2), 0)
        # ax.set_ylim(0, ax.get_ylim()[1] * (7 / 5))

        # prepend handles/labels so they appear at the top
        handles_pr, labels_pr = ax_pr.get_legend_handles_labels()
        handles = handles_pr + handles
        labels = labels_pr + labels

    # Put the legend outside the plot
    ax.legend(handles, labels, bbox_to_anchor=legend_loc, loc="upper left")

    # set formatting for xticks
    date_fmt = DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.tick_params(axis="x", rotation=30)
    if grid ==True:
        ax.grid()
        
    # # calculate metrics for data table underneath plot£
    # def calc_metric(metric) -> float:
    #     return y_sim.apply(metric, observed_array=y_obs)

    # metrs = pd.DataFrame(
    #     {
    #         "nse": calc_metric(metrics.nse),
    #         "kge_2009": calc_metric(metrics.kge_2009),
    #         "sa": calc_metric(metrics.sa),
    #         "me": calc_metric(metrics.me),
    #     }
    # )

    # # convert data in dataframe to strings
    # cell_text = [[f"{item:.2f}" for item in row[1]] for row in metrs.iterrows()]

    # table = ax_tbl.table(
    #     cellText=cell_text,
    #     rowLabels=metrs.index,
    #     colLabels=metrs.columns,
    #     loc="center",
    # )
    # ax_tbl.set_axis_off()

    # # give more vertical space in cells
    # table.scale(1, 1.5)

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)

    return fig, ax