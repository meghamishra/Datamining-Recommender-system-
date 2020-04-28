#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
#%%
df = pd.read_csv('recommendationMovie.csv', header = None)
data = np.asarray(df,dtype=int)

best_prob = get_best_probs(data)

#%%
#### Implementing various algorithms:

####################################### Thompson algorithms############################################

reward_sum_arr,regret=Thompson_sampling(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)


#%%
#### Implementing various algorithms:

####################################### Thompson algorithms FF############################################

reward_sum_arr,regret=Thompson_sampling_ff(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)

#%%
############################################## EXP3 algorithms#####################################

reward_sum_arr,regret=EXP3(data,best_prob)


### Plotting function:
plot_func(reward_sum_arr)

############################################E-Greedy#########################################
#%%

reward_sum_arr,regret=e_greedy(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)

#################################### E-Greedy full feedback algorithms##################################
#%%

reward_sum_arr,regret=e_greedy_ff(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)


############################################UCB Algorithm###################################################
#%%

reward_sum_arr,regret=UCB(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)




#######################################UCB full feedback Algorithm#######################################
#%%

reward_sum_arr,regret=UCB_ff(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)



####################################### Multi-weighted Algorithm######################################3
#%%

reward_sum_arr,regret=Multi_weighted(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)

####################################### Multi-weighted-full feedback Algorithm######################################3
#%%

reward_sum_arr,regret=Multi_weighted_ff(data,best_prob)

### Plotting function:
plot_func(reward_sum_arr)


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
