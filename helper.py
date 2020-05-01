import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# e_greedy_ff, 

def plot_func(reward_sum_arr,fname):
    plt.clf()
    plt.plot(reward_sum_arr)
    # plt.show()
    fname = "plots_q1/"+fname
    plt.savefig(fname)
    

def get_best_probs(data):
    reward = np.zeros(data.shape[0])
    best_prob = []
    for i in range(data.shape[1]):
        reward[data[:,i]==1] +=1
        prob_now = reward/(i+1)
        best_prob.append(max(prob_now))
    return best_prob

def get_gt_probs(data):
    cum_sums = np.cumsum(data,axis=1)
    gt_probs = cum_sums/np.arange(1,data.shape[1]+1)
    return gt_probs

def Thompson_sampling(data,best_prob,gt_probs):
    reward_sum = 0
    reward_sum_arr = []
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
        reward_sum+=data[result[t],t]
        reward_sum_arr.append(reward_sum)
        prob_now = (S[arm_now]+1)/(S[arm_now]+F[arm_now]+2)
        regret.append(best_prob[t]-gt_probs[arm_now,t])
    return reward_sum_arr,regret


def  Thompson_sampling_ff(data,best_prob,gt_probs):

    reward_sum = 0
    reward_sum_arr = []
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

        S[data[:,t]==1] +=1
        F[data[:,t]==0] +=1

        arm_now = result[t]
        reward_sum+=data[result[t],t]
        reward_sum_arr.append(reward_sum)
        prob_now = (S[arm_now]+1)/(S[arm_now]+F[arm_now]+2)
        regret.append(best_prob[t]-gt_probs[arm_now,t])

    return reward_sum_arr,regret

def EXP3(data,prob_best,gt_probs):

    lr = np.ones(data.shape[1])/(np.arange(1,data.shape[1]+1))**0.5
    Loss = np.zeros(data.shape[0])
    pr = np.ones(data.shape[0])/data.shape[0]
    selections = []
    loss_list = [0]
    loss_cum_avg_list = [0]
    regret = []
    reward_sum = 0
    reward_sum_arr = []
    for t in range(data.shape[1]):
        It = np.random.choice(np.arange(data.shape[0]),1,p = pr)
        selections.append(It)
        loss_list.append(1-data[It,t])
        loss_cum_avg_list.append(((1-data[It,t])+loss_cum_avg_list[-1]*t)/(t+1))
        Loss[It] += (1-data[It,t])/pr[It]
        regret.append(prob_best[t]-gt_probs[It,t])
        pr = np.exp(-lr[t]*Loss)/np.sum(np.exp(-lr[t]*Loss))
        reward_sum+= data[It,t]
        reward_sum_arr.extend(reward_sum)
    return reward_sum_arr,regret

def e_greedy_ff(data,prob_best,gt_probs):
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
        prob_now = reward_count[i]/arm_count[i]
        regret.append(prob_best[t]-gt_probs[i,t])
        reward = data[i,t]
        reward_count+=data[:,t]
        arm_count+=1
        arms.append(i)
        reward_sum += reward
        reward_sum_arr.append(reward_sum)
    return reward_sum_arr,regret


def UCB(data,prob_best,gt_probs):

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
            j=t
        else:
            UCB=mu+np.sqrt((2*np.log(t))/n)
            j=np.argmax(UCB)
            reward=data[j,t]
            reward_sum += reward
            n[j]+=1
            mu[j]=mu[j]+(1/n[j])*(reward-mu[j])
        arms.append(j)
        reward_sum_arr.append(reward_sum)
        regret.append(prob_best[t]-gt_probs[j,t])
    return reward_sum_arr,regret

def UCB_ff(data,best_prob,gt_probs):
    mu=np.zeros(data.shape[0])
    n=np.ones(data.shape[0])
    UCB=np.zeros(data.shape[0])
    arms = []
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    j=0
    for t in range(data.shape[1]):
        if (t<data.shape[0]):
            mu[t] = data[t,t]
            j=t
        else:
            UCB=mu+np.sqrt((2*np.log(t))/n)
            j=np.argmax(UCB)
            reward=data[j,t]
            reward_sum += reward
            n+=1
            mu=mu+(1/n)*(data[:,t]-mu)
        arms.append(j)
        reward_sum_arr.append(reward_sum)
        regret.append(best_prob[t]-gt_probs[j,t])
    return reward_sum_arr,regret


def Multi_weighted(data,prob_best,gt_probs):
    eta=1
    w=np.ones(data.shape[0])
    pr=np.ones(data.shape[0])
    armss = np.arange(data.shape[0])
    arm_selected = []
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    for t in range(data.shape[1]):
        pr=w/sum(w)
        arm=np.random.choice(armss,1,p=pr)
        reward = data[arm,t]
        loss=(1-data[arm,t])
        w[arm]=w[arm]*(1-eta*loss)
        eta = 1/np.sqrt(t+1)
        arm_selected.append(arm)
        reward_sum += reward
        reward_sum_arr.extend(reward_sum)
        regret.append(prob_best[t]-gt_probs[arm,t])
    return reward_sum_arr,regret

def Multi_weighted_ff(data,prob_best,gt_probs):

    eta=1
    w=np.ones(data.shape[0])
    pr=np.ones(data.shape[0])
    armss = np.arange(data.shape[0])
    arm_selected = []
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    for t in range(data.shape[1]):
        pr=w/sum(w)
        arm=np.random.choice(armss,1,p=pr)
        reward = data[arm,t]
        loss=(1-data[:,t])
        w=w*(1-eta*loss)
        eta = 1/np.sqrt(t+1)
        arm_selected.append(arm)
        reward_sum += reward
        reward_sum_arr.extend(reward_sum)
        regret.append(prob_best[t]-gt_probs[arm,t])
    return reward_sum_arr,regret

def EXP3_FF(data,prob_best,gt_probs):

    lr = np.ones(data.shape[1])/(np.arange(1,data.shape[1]+1))**0.5
    Loss = np.zeros(data.shape[0])
    pr = np.ones(data.shape[0])/data.shape[0]
    selections = []
    loss_list = [0]
    loss_cum_avg_list = [0]
    regret = []
    reward_sum = 0
    reward_sum_arr = []
    for t in range(data.shape[1]):

        It = np.random.choice(np.arange(data.shape[0]),1,p = pr)
        selections.append(It)
        loss_list.append(1-data[It,t])
        loss_cum_avg_list.append(((1-data[It,t])+loss_cum_avg_list[-1]*t)/(t+1))        
        Loss += (1-data[:,t])       
        regret.append(prob_best[t]-gt_probs[It,t])
        pr = np.exp(-lr[t]*Loss)/np.sum(np.exp(-lr[t]*Loss))        
        reward_sum+= data[It,t]
        reward_sum_arr.extend(reward_sum)
    return reward_sum_arr,regret

def e_greedy(data,prob_best,gt_probs):
    reward_count = np.zeros(data.shape[0])
    arm_count = np.zeros(data.shape[0])-0.001
    arms = []
    eps = 1
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    for t in range(data.shape[1]):
        temp=np.random.rand(1)
        # eps = 1/(t+1)
        eps = eps*0.995
        if temp<eps:
            i=np.random.randint(0,data.shape[0])
        else:
            mu = reward_count/arm_count               
            i=np.argmax(mu)
        prob_now = reward_count[i]/arm_count[i]
        regret.append(prob_best[t]-gt_probs[i,t])
        reward = data[i,t]
        reward_count[i]+=reward
        arm_count[i]+=1
        arms.append(i)
        reward_sum += reward
        reward_sum_arr.append(reward_sum)
    return reward_sum_arr,regret

def get_ten_best_movie_plots(data,best_prob,best_movies):
    eta=1
    w=np.ones(data.shape[0])
    pr=np.ones(data.shape[0])
    armss = np.arange(data.shape[0])
    arm_selected = []
    reward_sum = 0
    reward_sum_arr = []
    regret = []
    prob_best_movies = []
    for t in range(data.shape[1]):
        pr=w/sum(w)
        arm=np.random.choice(armss,1,p=pr)
        reward = data[arm,t]
        loss=(1-data[:,t])
        w=w*(1-eta*loss)
        eta = 1/np.sqrt(t+1)
        arm_selected.append(arm)
        reward_sum += reward
        reward_sum_arr.extend(reward_sum)
        regret.append(best_prob[t] - pr[arm])
        prob_best_movies.append(pr[best_movies])
    prob_best_movies_arr = np.asarray(prob_best_movies)
    for i in range(prob_best_movies_arr.shape[1]):
        plt.plot(prob_best_movies_arr[:,i])
    # plt.show()
    plt.savefig("plots_q1/ten_movies.png")