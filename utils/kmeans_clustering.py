#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:04:01 2023

@author: pwiersma
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
# from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.decomposition import PCA


def create_cluster_parsets(soilcalib_name, basin, prior_ranges, out_path, k= 4, plot = False):
    posterior_parset_file = join("/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/spotpy_outputs/",
                                        f"{soilcalib_name}_{basin} - Copy_posterior.csv")
    pars_df = pd.read_csv(posterior_parset_file,
                                header=0, index_col=0)
    pars_df.pop('KGE')
    # pars_dict = pars_df.to_dict(orient = 'index')
    
    minmaxes = prior_ranges
    pars_scaled = pd.DataFrame()
    for col in pars_df.columns:
        minimum,maximum = minmaxes[col]
        pars_scaled[col] = (pars_df[col] - minimum)/(maximum-minimum)
       
    
    kmeans = cluster.KMeans(n_clusters = k, random_state = 1)
        
    kmeans = kmeans.fit(pars_scaled)
    cluster_assignments = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    closest_param_sets = np.empty((k, pars_scaled.shape[1]))
    
    # Loop through each cluster
    for cluster_id in range(k):
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
        minimum,maximum = minmaxes[col]
        cluster_parsets[col] = (cluster_parsets_normed[col] * (maximum -minimum)) + minimum
    # cluster_parsets_dic = cluster_parsets.to_dict(orient = 'index')
    cluster_parsets.to_csv(out_path)
    
    N_axes = len(pars_scaled.columns)
    ## Ploting
    if plot ==True:
        f1,axes = plt.subplots(1,N_axes,figsize = (20,3),sharey = True)
        plt.suptitle(basin + ' normalized posterior parameters + ' +str(k)+' clusters', y = 1.1)
        for i,col in enumerate(pars_scaled.columns):
            ax = axes[i]
            pars_scaled[col].plot(ax = ax)
            ax.set_title(col)
            ax.set_ylim([0,1])
            
            twin = plt.twinx(ax)
            N = len(pars_scaled)
            x = np.arange(0,N,N/k)+ (N/k)/2
            # twin.scatter(x = x,y = cluster_centers[:,i], c= x, cmap = 'turbo')
            twin.scatter(x = x,y = closest_param_sets[:,i], c= x, cmap = 'turbo')
    
            twin.set_ylim([0,1])
            twin.set_yticks([])
            if i==0:
                ax.set_ylabel('Normalized parameter range')
            if i==int(N_axes/2):
                ax.set_xlabel('Best model runs')
        f1.savefig(f"/home/pwiersma/scratch/Figures/ewc_figures/SGM23/{basin}_cluster_parsets.svg",
                   bbox_inches = 'tight')
    return cluster_parsets






# # for basin in ['Jonschwil','Landquart','Landwasser','Rom','Verzasca']:
# for basin in ['Landwasser']:
#     posterior_parset_file = join("/home/pwiersma/scratch/Data/ewatercycle/experiments/wflow_julia_1000m_2005_2015_JLLRV/spotpy_outputs/",
#                                     f"ROPE_1000m_soilcalib_5000runs_2004_2015_KGE_{basin}_posterior.csv")
#     pars_df = pd.read_csv(posterior_parset_file,
#                                 header=0, index_col=0)
#     pars_df.pop('KGE')
    
    
    
#     # minmaxes = dict( KsatHorFrac = [1,300],
#     #                 f = [0.3,1.7],
#     #                 thetaR = [0.8,1.2],
#     #                 thetaS = [0.8,1.2],
#     #                 KsatVer = [0.8,1.2],
#     #                 SoilThickness = [0.8,1.2],
#     #                 InfiltCapSoil = [0.2,1.2],
#     #                 sfcf = [1.1,1.6],
#     #                 sfcf_scale = [1,2.5], 
#     #                 DD_min = [2,4] ,
#     #                 DD_max = [4,7], 
#     #                 mwf = [0,1],
#     #                 WHC = [0,0.2],
#     #                 TT = [-1,1],
#     #                 )
    
    
#     pars_scaled = pd.DataFrame()
#     for col in pars_df.columns:
#         minimum,maximum = prior_ranges[col]
#         pars_scaled[col] = (pars_df[col] - minimum)/(maximum-minimum)
        

    
#     # f1,axes = plt.subplots(1,15,figsize = (20,3),sharey = True)
#     # plt.suptitle(basin + ' normalized posterior parameters', y = 1.1)
#     # for i,col in enumerate(pars_scaled.columns):
#     #     pars_scaled[col].plot(ax = axes[i])
#     #     axes[i].set_title(col)
#     #     axes[i].set_ylim([0,1])
    
#     #%% k-means on all parameters (not PCA)
#     # wss = []
#     # silhouettes = []
#     # K = np.arange(2,12)
    
    
#     # scaler = MinMaxScaler()
#     # pars_scaled = scaler.fit_transform(pars)
    
#     # for k in K:
#     #     kmeans = cluster.KMeans(n_clusters = k)
        
#     #     kmeans = kmeans.fit(pars_scaled)
#     #     # Get cluster assignments for each parameter set
#     #     cluster_assignments = kmeans.labels_
        
#     #     # Get cluster centers (i.e., the centroids of each cluster)
#     #     cluster_centers = kmeans.cluster_centers_
        
#     #     wss_iter = kmeans.inertia_
#     #     wss.append(wss_iter)
#     #     labels=cluster.KMeans(n_clusters=k,random_state=200).fit(pars_scaled).labels_
#     #     silhouette = metrics.silhouette_score(pars_scaled,labels,metric="euclidean",sample_size=1000,random_state=200)
#     #     print ("Silhouette score for k(clusters) = "+str(k)+" is "+str(silhouette))
#     #     silhouettes.append(silhouette)
        
#     # # plt.figure()
#     # # plt.xlabel('K')
#     # # plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
#     # # plt.plot(K,wss)
#     # # plt.title(basin)
    
#     # plt.figure()
#     # plt.plot(K,silhouettes)
#     # plt.xlabel('K')
#     # plt.ylabel('Silhouette score')
#     # plt.title(basin)
#     # plt.ylim([0,1])

    
# #%%

#     pca = PCA(n_components = 2)
#     principalComponents = pca.fit_transform(pars_scaled)
#     pca_df = pd.DataFrame(data = principalComponents, columns = ['pca1','pca2'])
    
#     wss = []
#     silhouettes = []
#     K = np.arange(2,12)
#     for k in K:
#         kmeans = cluster.KMeans(n_clusters = k)
        
#         kmeans = kmeans.fit(pca_df)
#         # Get cluster assignments for each parameter set
#         cluster_assignments = kmeans.labels_
        
#         # Get cluster centers (i.e., the centroids of each cluster)
#         cluster_centers = kmeans.cluster_centers_
        
#         wss_iter = kmeans.inertia_
#         wss.append(wss_iter)
#         labels=cluster.KMeans(n_clusters=k,random_state=200).fit(pca_df).labels_
#         silhouette = metrics.silhouette_score(pca_df,labels,metric="euclidean",sample_size=1000,random_state=200)
#         print ("Silhouette score for k(clusters) = "+str(k)+" is "+str(silhouette))
#         silhouettes.append(silhouette)
        
    
#     # kmeans = cluster.KMeans(n_clusters = K[np.argmax(silhouettes)])
#     k = 4
#     kmeans = cluster.KMeans(n_clusters = k)
        
#     kmeans = kmeans.fit(pca_df)
    
#     pca_df['Clusters'] = kmeans.labels_
    
#     f1,(ax1,ax2) = plt.subplots(1,2, figsize = (12,4))
#     plt.suptitle(basin, y = 1.05)
#     ax1.plot(K,silhouettes)
#     ax1.set_xlabel('K')
#     ax1.set_ylabel('Silhouette score')
#     ax1.set_title('silhouette score [-1:1]')
#     ax1.set_ylim([0,1])
    
#     sns.scatterplot(x="pca1", y="pca2",hue = 'Clusters',  
#                     data=pca_df,palette='tab10',legend = 'full',
#                     ax = ax2)
    
#     ax2.set_title('k-means clusters')
#     # plt.savefig(f"/home/pwiersma/scratch/Figures/ewc_figures/PCA/{basin}_PCA_{k}_clusters.png",
#     #             bbox_inches = 'tight')
    
#     #%% Clustering on original data
#     wss = []
#     silhouettes = []
#     K = np.arange(2,12)
    
#     for k in K:
#         kmeans = cluster.KMeans(n_clusters = k)
        
#         kmeans = kmeans.fit(pars_scaled)
#         # Get cluster assignments for each parameter set
#         cluster_assignments = kmeans.labels_
        
#         # Get cluster centers (i.e., the centroids of each cluster)
#         cluster_centers = kmeans.cluster_centers_
        
#         wss_iter = kmeans.inertia_
#         wss.append(wss_iter)
#         labels=cluster.KMeans(n_clusters=k,random_state=200).fit(pars_scaled).labels_
#         silhouette = metrics.silhouette_score(pars_scaled,labels,metric="euclidean",sample_size=1000,random_state=200)
#         print ("Silhouette score for k(clusters) = "+str(k)+" is "+str(silhouette))
#         silhouettes.append(silhouette)
        
#     f1,ax1 = plt.subplots()
#     ax1.plot(K,silhouettes)
#     ax1.set_xlabel('K')
#     ax1.set_ylabel('Silhouette score')
#     ax1.set_title('silhouette score [-1:1]')
#     ax1.set_ylim([0,1])
    
#     k = 6
#     kmeans = cluster.KMeans(n_clusters = k)
        
#     kmeans = kmeans.fit(pars_scaled)
#     cluster_assignments = kmeans.labels_
#     cluster_centers = kmeans.cluster_centers_
    
#     closest_param_sets = np.empty((k, pars_scaled.shape[1]))
    
#     # Loop through each cluster
#     for cluster_id in range(k):
#         # Find the indices of parameter sets in the current cluster
#         cluster_indices = np.where(cluster_assignments == cluster_id)[0]
    
#         # Calculate distances from all parameter sets in the cluster to the centroid
#         distances = np.linalg.norm(pars_scaled.values[cluster_indices] - cluster_centers[cluster_id], axis=1)
    
#         # Find the index of the parameter set with the minimum distance
#         closest_index = cluster_indices[np.argmin(distances)]
    
#         # Store the closest parameter set
#         closest_param_sets[cluster_id] = pars_scaled.values[closest_index]
#     #%%
#     f1,axes = plt.subplots(1,15,figsize = (20,3),sharey = True)
#     plt.suptitle(basin + ' normalized posterior parameters', y = 1.1)
#     for i,col in enumerate(pars_scaled.columns):
#         ax = axes[i]
#         pars_scaled[col].plot(ax = ax)
#         ax.set_title(col)
#         ax.set_ylim([0,1])
        
#         twin = plt.twinx(ax)
#         N = len(pars_scaled)
#         x = np.arange(0,N,N/k)+ (N/k)/2
#         # twin.scatter(x = x,y = cluster_centers[:,i], c= x, cmap = 'turbo')
#         twin.scatter(x = x,y = closest_param_sets[:,i], c= x, cmap = 'turbo')

#         twin.set_ylim([0,1])
#         twin.set_yticks([])
#         # sns.scatterplot(ax = twin, x = x, y = cluster_centers[:,i],
#         #                 hue = x,
#         #                 palette = 'tab10')