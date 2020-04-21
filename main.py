# Run interactive before sharing

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import *
#%%
df = pd.read_csv('recommendationMovie.csv', header = None)
data = np.asarray(df,dtype=int)

# row = movie, column = day
# for each arm set Si, Fi = 0
# for each t
#   for each arm sample from beta
#   play i = argmax(samples), observe reward
#   if r = 1, Si +=1, else Fi +=1

# THOMPSON SAMPLING

#%%
# -----------CODE FOR THOMPSON------------LINE-23-50----
# S = np.zeros(data.shape[0])
# F = np.zeros(data.shape[0])
# theta = np.zeros(data.shape[0])
# result = -1*np.ones(data.shape[1],dtype=int)
# #%%
# for t in range(data.shape[1]):
#     if t%100==0:
#         print(t)
#     for arm in range(data.shape[0]):
#         theta[arm] = np.random.beta(S[arm]+1,F[arm]+1)
#     result[t] = np.argmax(theta)
#     if data[result[t],t] ==1:
#         S[result[t]] +=1
#     else:
#         F[result[t]] +=1
    


# #row movie, column is time (day)
# #%%
# u = (S+1)/(S+F+2)
# regret = max(u)-u[result]
# regret_by_t = []
# for t in range(data.shape[1]):
#     regret_t = np.sum(regret[:t])
#     regret_by_t.append(regret_t/(t+1)) 
    
# plt.plot(regret_by_t)

# ----------------THOMPSON ENDS
#EXP3

#%%
#eta= non decreasing sequence of N real numbers 
# p1 = uniform distribution from {1 to k}
# for t in 1...n
# draw arm It with probability pt
#for each arm i, estimated loss l'i,t= lit/pit (when It=i)
# and update cummulative loss= L'i,t=L'i,t-1+l'i,s
#compute new probability distribution over arm pt+1=(p1,t+1....pk,t+1),
#where pi,t+1=exp(-eta t L'i,t)/sum(exp(-eta t L'i,t))


# %%
# %%
np.arange(1,10)
# %%
# -------------------CODE FOR EXP3-------- LINE 71-97----
# lr = np.ones(data.shape[1])/(np.arange(1,data.shape[1]+1))**0.5
# Loss = np.zeros(data.shape[0])
# pr = np.ones(data.shape[0])/data.shape[0]
# selections = []
# loss_list = [0]
# loss_cum_avg_list = [0]
# for t in range(data.shape[1]):
#     It = np.random.choice(np.arange(data.shape[0]),1,p = pr)
#     selections.append(It)
#     loss_list.append(1-data[It,t])
#     loss_cum_avg_list.append(((1-data[It,t])+loss_cum_avg_list[-1]*t)/(t+1))
#     Loss[It] += (1-data[It,t])/pr[It]
#     pr = np.exp(-lr[t]*Loss)/np.sum(np.exp(-lr[t]*Loss))


# # %%

# temp = np.array(loss_list).flatten()
# loss_cum_avg_list = loss_cum_avg_list[1:]
# plt.plot(loss_cum_avg_list)
# # loss_list_cumm=np.cumsum(loss_list)
# # e-greedy!!!
# # UCB!!!
# # plt.plot(loss_list_cumm/np.arange(1,data.shape[1]+1))

# # %%
# plt.plot(loss_list_cumm)

# UCB
# set mu_i (mean reward for arm i),ni (num of times arm i picked) to zeros
# Try all once, ni = 1
# for t-----
# calc UCB ui + root(2lnt/ni) 
# j = argmax (UCB)
# nj+=1, mu_j = mu_j+1/nj(yt-mu_j)

# %%
# ----------CODE FOR UCB------LINE 110-129---------

# mu=np.zeros(data.shape[0])
# n=np.ones(data.shape[0])
# UCB=np.zeros(data.shape[0])
# arms = []
# reward_sum = 0
# reward_sum_arr = []
# for t in range(data.shape[1]):
#     if (t<data.shape[0]):
#         mu[t] = data[t,t]
#     else:
#         UCB=mu+np.sqrt((2*np.log(t))/n)
#         j=np.argmax(UCB)
#         reward=data[j,t]
#         reward_sum += reward
#         n[j]+=1
#         mu[j]=mu[j]+(1/n[j])*(reward-mu[j])
#         arms.append(j)
#         reward_sum_arr.append(reward_sum)

# # plt.plot(reward_sum_arr)
        

# %%
# e-greedy LINE 134-154
# reward_count = np.zeros(data.shape[0])
# arm_count = np.zeros(data.shape[0])-0.001
# arms = []
# eps = 1
# reward_sum = 0
# reward_sum_arr = []
# for t in range(data.shape[1]):
#     temp=np.random.rand(1)
#     # eps = 1/(t+1)
#     eps = eps*0.995
#     if temp<eps:
#         i=np.random.randint(0,data.shape[0])
#     else:
#         mu = reward_count/arm_count               
#         i=np.argmax(mu)
#     reward = data[i,t]
#     reward_count[i]+=reward
#     arm_count[i]+=1
#     arms.append(i)
#     reward_sum += reward
#     reward_sum_arr.append(reward_sum)





# %%
# # --------------CODE FOR MULT-weightUpdate WITH FULL FEEDBACK--------
# eta=1
# w=np.ones(data.shape[0])
# pr=np.ones(data.shape[0])
# armss = np.arange(data.shape[0])
# arm_selected = []
# reward_sum = 0
# reward_sum_arr = []
# for t in range(data.shape[1]):
#     pr=w/sum(w)
#     arm=np.random.choice(armss,1,p=pr)
#     reward = data[arm,t]
#     loss=(1-data[:,t])
#     w=w*(1-eta*loss)
#     eta = 1/np.sqrt(t+1)
#     arm_selected.append(arm)
#     reward_sum += reward
#     # print(reward_sum)
#     reward_sum_arr.extend(reward_sum)


#%%
# # ------

# %%
# %%
# ----------CODE FOR UCB FULL FEEDBACK------LINE 110-129---------

# mu=np.zeros(data.shape[0])
# n=np.ones(data.shape[0])
# UCB=np.zeros(data.shape[0])
# arms = []
# reward_sum = 0
# reward_sum_arr = []
# for t in range(data.shape[1]):
#     if (t<data.shape[0]):
#         mu[t] = data[t,t]
#     else:
#         UCB=mu+np.sqrt((2*np.log(t))/n)
#         j=np.argmax(UCB)
#         reward=data[j,t]
#         reward_sum += reward
#         n+=1
#         mu=mu+(1/n)*(data[:,t]-mu)
#         arms.append(j)
#         reward_sum_arr.append(reward_sum)

# plt.plot(reward_sum_arr)

# %%
# ---------e-greedy with full feedback---------------
reward_count = np.zeros(data.shape[0])
arm_count = np.zeros(data.shape[0])-0.001
arms = []
eps = 1
reward_sum = 0
reward_sum_arr = []
for t in range(data.shape[1]):
    temp=np.random.rand(1)
    # eps = 1/(t+1)
    eps = eps*0.995
    if temp<eps:
        i=np.random.randint(0,data.shape[0])
    else:
        mu = reward_count/arm_count               
        i=np.argmax(mu)

    reward = data[i,t]
    # reward_count[i]+=reward
    # arm_count[i]+=1
    reward_count+=data[:,t]
    arm_count+=1
    arms.append(i)
    reward_sum += reward
    reward_sum_arr.append(reward_sum)

# plt.plot(reward_sum_arr)



# %%
# def get_best_probs(data):
#     reward = np.zeros(data.shape[0])
#     best_prob = []
#     for i in range(data.shape[1]):
#         reward[data[:,i]==1] +=1
#         prob_now = reward/(i+1)
#         best_prob.append(max(prob_now))
#     return best_prob

# best_probs = get_best_probs(data)
# %%
