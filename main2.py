#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
#%%
df = pd.read_csv('recommendationMovie.csv', header = None)
data = np.asarray(df,dtype=int)

best_prob = get_best_probs(data)
gt_probs = get_gt_probs(data)

#%%
#### Implementing various algorithms:

####################################### Thompson algorithms############################################

reward_sum_arr,regret=Thompson_sampling(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "Thomp_loss.png"
plot_func(cum_loss,fname)

fname = "Thomp_regret.png"
plot_func(cum_regret,fname)


#%%
#### Implementing various algorithms:

####################################### Thompson algorithms FF############################################

reward_sum_arr,regret=Thompson_sampling_ff(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "Thomp_ff_loss.png"
plot_func(cum_loss,fname)

fname = "Thomp_ff_regret.png"
plot_func(cum_regret,fname)#%%
############################################## EXP3 algorithms#####################################
#%%
reward_sum_arr,regret=EXP3(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "EXP_loss.png"
plot_func(cum_loss,fname)

fname = "EXP_regret.png"
plot_func(cum_regret,fname)
############################################## EXP3 algorithms  Full Feebback#####################
#%%
reward_sum_arr,regret=EXP3_FF(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "EXP_ff_loss.png"
plot_func(cum_loss,fname)

fname = "EXP_ff_regret.png"
plot_func(cum_regret,fname)
############################################E-Greedy#########################################
#%%

reward_sum_arr,regret=e_greedy(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "eGreedy_loss.png"
plot_func(cum_loss,fname)

fname = "eGreedy_regret.png"
plot_func(cum_regret,fname)
#################################### E-Greedy full feedback algorithms##################################
#%%

reward_sum_arr,regret=e_greedy_ff(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "eGreedy_ff_loss.png"
plot_func(cum_loss,fname)

fname = "eGreedy_ff_regret.png"
plot_func(cum_regret,fname)

############################################UCB Algorithm###################################################
#%%

reward_sum_arr,regret=UCB(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "UCB_loss.png"
plot_func(cum_loss,fname)

fname = "UCB_regret.png"
plot_func(cum_regret,fname)


#######################################UCB full feedback Algorithm#######################################
#%%

reward_sum_arr,regret=UCB_ff(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "UCB_ff_loss.png"
plot_func(cum_loss,fname)

fname = "UCB_ff_regret.png"
plot_func(cum_regret,fname)


####################################### Multi-weighted Algorithm######################################3
#%%

reward_sum_arr,regret=Multi_weighted(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "mulW_loss.png"
plot_func(cum_loss,fname)

fname = "mulW_regret.png"
plot_func(cum_regret,fname)
####################################### Multi-weighted-full feedback Algorithm######################################3
#%%

reward_sum_arr,regret=Multi_weighted_ff(data,best_prob,gt_probs)
cum_loss = np.arange(1,len(reward_sum_arr)+1) - reward_sum_arr
cum_regret = np.cumsum(regret)
### Plotting function:
fname = "mulW_ff_loss.png"
plot_func(cum_loss,fname)

fname = "mulW_ff_regret.png"
plot_func(cum_regret,fname)

#%%
rewards = np.sum(data,axis=1)
sorted_reward_args = np.argsort(rewards)
print(sorted_reward_args[-10:])
#BEST MOVIE INDEX: 806 778  15 211 406 160 968 887 328 107
# %%
best_movies = [806, 778,  15, 211, 406, 160, 968, 887, 328, 107]
get_ten_best_movie_plots(data,best_prob,best_movies)


# return reward_sum_arr,regret


# %%

     

# %%
