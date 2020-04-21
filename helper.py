import numpy as np
import pandas as pd

# e_greedy_ff, 

def get_best_probs(data):
    reward = np.zeros(data.shape[0])
    best_prob = []
    for i in range(data.shape[1]):
        reward[data[:,i]==1] +=1
        prob_now = reward/(i+1)
        best_prob.append(max(prob_now))
    return best_prob


def Thompson_sampling(data,best_prob):

    S = np.zeros(data.shape[0])
    F = np.zeros(data.shape[0])
    theta = np.zeros(data.shape[0])
    result = -1*np.ones(data.shape[1],dtype=int)
    regret = []
    for t in range(data.shape[1]):
        if t%100==0:
            print(t)
        for arm in range(data.shape[0]):
            theta[arm] = np.random.beta(S[arm]+1,F[arm]+1)
        result[t] = np.argmax(theta)
        if data[result[t],t] ==1:
            S[result[t]] +=1
        else:
            F[result[t]] +=1
        arm_now = result[t]
        prob_now = (S[arm_now]+1)/(S[arm_now]+F[arm_now]+2)
        regret.append(best_prob[t]-prob_now)
    return F,S



def EXP3(data):

    lr = np.ones(data.shape[1])/(np.arange(1,data.shape[1]+1))**0.5
    Loss = np.zeros(data.shape[0])
    pr = np.ones(data.shape[0])/data.shape[0]
    selections = []
    loss_list = [0]
    loss_cum_avg_list = [0]
    regret = []
    for t in range(data.shape[1]):
        It = np.random.choice(np.arange(data.shape[0]),1,p = pr)
        selections.append(It)
        loss_list.append(1-data[It,t])
        loss_cum_avg_list.append(((1-data[It,t])+loss_cum_avg_list[-1]*t)/(t+1))
        Loss[It] += (1-data[It,t])/pr[It]
        regret.append(prob_best[t]-pr[It])
        pr = np.exp(-lr[t]*Loss)/np.sum(np.exp(-lr[t]*Loss))
        
    return loss_cum_avg_list,regret

def e_greedy_ff(data):
    reward_count = np.zeros(data.shape[0])
    arm_count = np.zeros(data.shape[0])-0.001
    arms = []
    eps = 1
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    for t in range(data.shape[1]):
        temp=np.random.rand(1)
        eps = eps*0.995
        if temp<eps:
            i=np.random.randint(0,data.shape[0])
        else:
            mu = reward_count/arm_count               
            i=np.argmax(mu)

        reward = data[i,t]
        reward_count+=data[:,t]
        arm_count+=1
        arms.append(i)
        reward_sum += reward
        reward_sum_arr.append(reward_sum)
    return reward_sum_arr


def UCB(data):
    # set mu_i (mean reward for arm i),ni (num of times arm i picked) to zeros
    # Try all once, ni = 1
    # for t-----
    # calc UCB ui + root(2lnt/ni) 
    # j = argmax (UCB)
    # nj+=1, mu_j = mu_j+1/nj(yt-mu_j)


    mu=np.zeros(data.shape[0])
    n=np.ones(data.shape[0])
    UCB=np.zeros(data.shape[0])
    arms = []
    reward_sum = 0
    reward_sum_arr = []
    for t in range(data.shape[1]):
        if (t<data.shape[0]):
            mu[t] = data[t,t]
        else:
            UCB=mu+np.sqrt((2*np.log(t))/n)
            j=np.argmax(UCB)
            reward=data[j,t]
            reward_sum += reward
            n[j]+=1
            mu[j]=mu[j]+(1/n[j])*(reward-mu[j])
            arms.append(j)
            reward_sum_arr.append(reward_sum)
            
    return reward_sum_arr

def UCB_ff(data,best_prob):
    mu=np.zeros(data.shape[0])
    n=np.ones(data.shape[0])
    UCB=np.zeros(data.shape[0])
    arms = []
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    for t in range(data.shape[1]):
        if (t<data.shape[0]):
            mu[t] = data[t,t]
        else:
            UCB=mu+np.sqrt((2*np.log(t))/n)
            j=np.argmax(UCB)
            reward=data[j,t]
            reward_sum += reward
            n+=1
            mu=mu+(1/n)*(data[:,t]-mu)
            arms.append(j)
            reward_sum_arr.append(reward_sum)
        regret.append(best_prob[t] - mu[j])
    return reward_sum_arr,regret


def Multi_weighted(data):

    eta=1
    w=np.ones(data.shape[0])
    pr=np.ones(data.shape[0])
    armss = np.arange(data.shape[0])
    arm_selected = []
    reward_sum = 0
    reward_sum_arr = []
    for t in range(data.shape[1]):
        pr=w/sum(w)
        arm=np.random.choice(armss,1,p=pr)
        reward = data[arm,t]
        loss=(1-data[:,t])
        w=w*(1-eta*loss)
        eta = 1/np.sqrt(t+1)
        arm_selected.append(arm)
        reward_sum += reward
        # print(reward_sum)
        reward_sum_arr.extend(reward_sum)
    return reward_sum_arr,arm_selected