# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:29:28 2020

@author: megha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_Q2 import *
pd.set_option('display.max_columns', 500)



########## DEFINING HYPER PARAMETERS ##########################################

batch_size = 100
iters = 100
lr = 0.01
selection=2 # 1 for random initialization and 2 for k-means++ initialization

############################# Reading dataset #################################
df=pd.read_csv(r"Movies.csv")

#######################Getting data in required format ########################
cleaned_arr=np.array(cleaning_data(df))


###################### Implementing  K-means algorithm #########################
num_clus=10
centers_ind = np.random.randint(0,len(cleaned_arr),num_clus)    
centroids=cleaned_arr[centers_ind] 
centroids=k_means_centriods(num_clus,cleaned_arr,iters,batch_size,lr,centroids)
clusters=make_clusters(cleaned_arr,centroids,num_clus)

### Plotting clusters###

min_max_mean_plot(clusters,centroids,num_clus)


###################### Implementing  K-means++ algorithm #########################
num_clus=10  
centroids= k_means_plus_centriods_initialization(num_clus,cleaned_arr)
centroids=k_means_centriods(num_clus,cleaned_arr,iters,batch_size,lr,centroids)
clusters=make_clusters(cleaned_arr,centroids,num_clus)

### Plotting clusters###

min_max_mean_plot(clusters,centroids,num_clus)

###################### K-MEANS WITH curves #####################
clus_values=[5,20,100,500]

plot_curves_k_means(clus_values,cleaned_arr,iters,batch_size,lr,selection)

        
    
