#%%
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)


# %%
df=pd.read_csv("Movies.csv")
df_dum=pd.get_dummies(df['titleType'])
df=pd.merge(df,df_dum,left_index=True,right_index=True)

#CHECK WEIGHTAGE FOR YEAR RUNTIME LATER
df['startYear']=df['startYear']-min(df['startYear'])
df_curated=df.drop(['tconst','titleType','originalTitle'],axis=1)
df_curated['startYear']=df_curated['startYear']/df_curated['startYear'].max()
df_curated['runtimeMinutes']=(df_curated['runtimeMinutes']-df_curated['runtimeMinutes'].min())/(df_curated['runtimeMinutes'].max()-df_curated['runtimeMinutes'].min())

# %%
num_clus=10
batch_size = 100
iters = 100
lr = 0.01
#%%
arr=np.array(df_curated)
centers_ind = np.random.randint(0,len(arr),num_clus)
centroids=arr[centers_ind]

#%%
trial_batch = arr[np.random.randint(0,len(arr),batch_size)]

# %%

# plt.plot(elbow_value)
        

        
        

#%% SAUMYA ZONE, CAREFULLY PROCEED
print(centroids.shape)

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
        centroids[i] = centroids[i] - lr*(np.mean(mini_batch[closest_center_arg==i],axis = 0)-centroids[i])







# %% 

elbow_values=[]

for num_clus in [10,20,30]:
    centers_ind = np.random.randint(0,len(arr),num_clus)
    centroids=arr[centers_ind]
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
            centroids[i] = centroids[i] - lr*(np.mean(mini_batch[closest_center_arg==i],axis = 0)-centroids[i])


    clusters=[[] for i in range(num_clus)]
    for num in arr:
        dist=np.sum((num-centroids)**2,axis=1)
        cluster_value=np.argmin(dist)
        clusters[cluster_value].append(num)

    sum_val=0
    for c in range(len(clusters)):
        dist=(np.sum((clusters[c]-centroids[c])**2,axis=1)).sum()

    elbow_values.apend(dist)

# %%  K-means++
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







#%% Saumya's temp
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = a.reshape(a.shape[0],a.shape[1],1)
a = a.swapaxes(0,2)
a = np.repeat(a,2,axis = 0)
print(a[1,:,1])


# %%

