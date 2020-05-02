## Final

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_Q2 import *
pd.set_option('display.max_columns', 500)



########## DEFINING HYPER PARAMETERS ##########################################

batch_size = 1000
iters = 1000
lr = 0.01
reduced=2 # 1 for PCA reduced data and 2 for original data

############################# Reading dataset #################################
df=pd.read_csv(r"Movies.csv")

#######################Getting data in required format ########################
cleaned_arr=cleaning_data(df,reduced)

###### Implementing  K-means algorithm with Random Initialization##############
num_clus=10
centers_ind = np.random.randint(0,len(cleaned_arr),num_clus)    
centroids=cleaned_arr[centers_ind] 
centroids=k_means_centriods(num_clus,cleaned_arr,iters,batch_size,lr,centroids)
clusters,labels=make_clusters(cleaned_arr,centroids,num_clus)

### Plotting clusters###
print("clusters obtained for K-means with Random Initialization")
cluster_plot(cleaned_arr,labels,reduced)


###### Implementing  K-means algorithm with Kmeans++ Initialization ###########
num_clus=10  
centroids= k_means_plus_centriods_initialization(num_clus,cleaned_arr)
centroids=k_means_centriods(num_clus,cleaned_arr,iters,batch_size,lr,centroids)
clusters,labels=make_clusters(cleaned_arr,centroids,num_clus)

### Plotting clusters###
print("clusters obtained for K-means with Kmeans++ Initialization")
cluster_plot(cleaned_arr,labels,reduced)

###################### K-MEANS WITH CURVES ####################################
clus_values=[5,20,50,100,300,400,500]
selection=1 # 1 for random initialization and 2 for k-means++ initialization
plot_curves_k_means(clus_values,cleaned_arr,iters,batch_size,lr,selection)

        
    
