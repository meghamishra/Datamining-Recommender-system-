# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:33:35 2020

@author: megha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cleaning_data(df):
    
    df_dum=pd.get_dummies(df['titleType'])
    df=pd.merge(df,df_dum,left_index=True,right_index=True)
    df['startYear']=df['startYear']-min(df['startYear'])   
    df_curated=df.drop(['tconst','titleType','originalTitle'],axis=1)    
    df_curated['startYear']=df_curated['startYear']/df_curated['startYear'].max()    
    df_curated['runtimeMinutes']=(df_curated['runtimeMinutes']-df_curated['runtimeMinutes'].min())/(df_curated['runtimeMinutes'].max()-df_curated['runtimeMinutes'].min())
    
    return df_curated

def k_means_plus_centriods_initialization(num_clus,arr):
    centers = []
    i1 = np.random.randint(0,arr.shape[0])
    # num_clus = 4
    centers.append(arr[i1])
    for c in range(num_clus-1):
        sum_dists = 0
        for cen in centers:
            dists = np.sum((arr-np.array(cen))**2,axis = 1)
            sum_dists += dists
        sum_dists_norm = sum_dists/np.sum(sum_dists)
        idx = np.random.choice(np.arange(0,arr.shape[0]),1,p = sum_dists_norm)
        centers.append(arr[idx])
        
    centroids=np.vstack(np.array(centers))
    return centroids
    
def k_means_centriods(num_clus,arr,iters,batch_size,lr,centroids):  

    for itrs in range(iters):
        centroid_arr = centroids.reshape((centroids.shape[0],centroids.shape[1],1))
        centroid_arr = centroid_arr.swapaxes(0,2)    
        centroid_arr = np.repeat(centroid_arr,batch_size,axis=0)
        data_idx = np.random.randint(0,len(arr),batch_size)
        mini_batch = arr[data_idx]
        data_array = mini_batch.reshape((mini_batch.shape[0],mini_batch.shape[1],1))
        data_array = data_array.repeat(centroid_arr.shape[2],axis = 2)
        dists = np.sum((data_array-centroid_arr)**2,axis = 1)
        closest_center_arg = np.argmin(dists,axis = 1)
        
    
        
        for i in range(num_clus):   
            if (closest_center_arg==i).any(): 
                centroids[i] = centroids[i] - lr*(np.mean(mini_batch[closest_center_arg==i],axis = 0)-centroids[i])
                
                
    return centroids


def make_clusters(arr,centroids,num_clus):
        clusters=[[] for i in range(num_clus)]
        for num in arr:
            dist=np.sum((num-centroids)**2,axis=1)
            cluster_value=np.argmin(dist)
            clusters[cluster_value].append(num)
            
        return clusters
    
def min_max_mean_plot(clusters,centroids,num_clus):
    plot_mean=[]
    plot_min=[]
    plot_max=[]   
    for c in range(num_clus): 
        sum_mean=0
        if len(clusters[c])>0:
            plot_mean.append(sum((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5)/len(clusters[c]))
            plot_max.append(max((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5))
            plot_min.append(min((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5))
           
    print('plots for Cluster value '+str(num_clus) )    
    print("mean value plot")
    plt.plot(plot_mean)
    plt.show()
    print("max value plot")
    plt.plot(plot_max)
    plt.show()
    print("min value plot")
    plt.plot(plot_min)
    plt.show()
    
    
def plot_curves_k_means(clus_values,arr,iters,batch_size,lr,selection): 
    elbow_value=[]
    
    for num_clus in clus_values: 
          
        plot_mean=[]
        plot_min=[]
        plot_max=[]
        
        if selection==1:
            centers_ind = np.random.randint(0,len(arr),num_clus)    
            centroids=arr[centers_ind] 
        else:
            centroids= k_means_plus_centriods_initialization(num_clus,arr)
            
        centroids=k_means_centriods(num_clus,arr,iters,batch_size,lr,centroids)
        clusters=make_clusters(arr,centroids,num_clus)
        sum_val=0
        for c in range(num_clus): 
            sum_mean=0
            if len(clusters[c])>0:
                sum_val=sum_val+sum((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5)
                plot_mean.append(sum((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5)/len(clusters[c]))
                plot_max.append(max((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5))
                plot_min.append(min((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5))
               
        print('plots for Cluster value '+str(num_clus) )    
        print("mean value plot")
        plt.plot(plot_mean)
        plt.show()
        print("max value plot")
        plt.plot(plot_max)
        plt.show()
        print("min value plot")
        plt.plot(plot_min)
        plt.show()
    
        elbow_value.append(sum_val)    
    
    print("elbow curve")
    plt.plot(clus_values,elbow_value)
    plt.show()
        
        


            

    
    
