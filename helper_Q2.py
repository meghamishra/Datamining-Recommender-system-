## Final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler


def cleaning_data(df,reduced):
    
    df_dum=pd.get_dummies(df['titleType'])
    df=pd.merge(df,df_dum,left_index=True,right_index=True)
    df['startYear']=df['startYear']-min(df['startYear'])   
    df_curated=df.drop(['tconst','titleType','originalTitle'],axis=1)    
    df_curated['startYear']=df_curated['startYear']/df_curated['startYear'].max()    
    df_curated['runtimeMinutes']=(df_curated['runtimeMinutes']-df_curated['runtimeMinutes'].min())/(df_curated['runtimeMinutes'].max()-df_curated['runtimeMinutes'].min())
    cleaned_arr=np.array(df_curated)
    if reduced==1:
        
        cleaned_arr=cleaned_arr-np.mean(cleaned_arr,axis=0)
        pca=PCA(n_components=20)
        cleaned_arr=pca.fit_transform(cleaned_arr)
    
    return cleaned_arr
def k_means_plus_centriods_initialization(num_clus,arr):
    centers = []
    i1 = np.random.randint(0,arr.shape[0])

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
                centroids[i] = centroids[i] + lr*(np.mean(mini_batch[closest_center_arg==i],axis = 0)-centroids[i])
                
                
    return centroids


def make_clusters(arr,centroids,num_clus):
        clusters=[[] for i in range(num_clus)]
        labels=[]
        for num in arr:
            dist=np.sum((num-centroids)**2,axis=1)
            
            cluster_value=np.argmin(dist)
           
            clusters[cluster_value].append(num)
            labels.append(cluster_value)
        return clusters,labels
    
def cluster_plot(arr,labels_arr,reduced):
    if reduced==2:
        ssfit=arr-np.mean(arr,axis=0)
        pca=PCA(n_components=2)
        arr=pca.fit_transform(ssfit)

    plt.scatter(arr[:,0],arr[:,1],c=labels_arr)
    plt.xlabel('PCA_Component1')
    plt.ylabel('PCA_Component2')
    plt.show() 
    

   
def plot_curves_k_means(clus_values,arr,iters,batch_size,lr,selection): 
    elbow_value=[]
    min_value=[]
    max_value=[]
    mean_value=[]
    
    for num_clus in clus_values: 
        plot_mean=[]
             
        if selection==1:
            centers_ind = np.random.randint(0,len(arr),num_clus)    
            centroids=arr[centers_ind] 
            str_title="K-means"
        else:
            centroids= k_means_plus_centriods_initialization(num_clus,arr)
            str_title="K-means++"
            
        centroids=k_means_centriods(num_clus,arr,iters,batch_size,lr,centroids)
        clusters,labels=make_clusters(arr,centroids,num_clus)        
        sum_val=0
        
        for c in range(num_clus): 
            if len(clusters[c])>0:
    
                dist_value=sum((np.sum((clusters[c]-centroids[c])**2,axis=1))**0.5)
                sum_val=sum_val+dist_value
                plot_mean.append(dist_value/len(clusters[c]))
            else:
                plot_mean.append(0)


        elbow_value.append(sum_val)   
        min_value.append(min(plot_mean))  
        max_value.append(max(plot_mean))
        mean_value.append(np.mean(plot_mean))

    print("elbow curve")
    plt.plot(clus_values,elbow_value)
    plt.xlabel('number of clusters')
    plt.ylabel('intra cluster distance')
    plt.title(str_title)
    plt.show()

    plt.plot(clus_values,mean_value,label='mean')
    plt.plot(clus_values,max_value,label='max')
    plt.plot(clus_values,min_value,label='min') 
    plt.legend()
    plt.xlabel('number of clusters')
    plt.ylabel('distance values')
    plt.title(str_title)   
    
    plt.show()
        

            

    
    
