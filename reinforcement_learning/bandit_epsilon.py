import numpy as np
from utils.epsilon import standard_epsilon, decaying_epsilon, exponential_decay_epsilon
import matplotlib.pyplot as plt

class bandit:
    def __init__(self, win_rate):
        self.win_rate = win_rate
        self.n = 0
        self.sample_mean = 0
        
    def pull(self):
        return np.random.random() < self.win_rate
        
    def update(self, reward):
        self.n += 1
        learning_rate = 1 / self.n
        self.sample_mean = self.sample_mean + learning_rate * (reward - self.sample_mean)

if __name__ == '__main__':
    min_epsilon = 0.05
    init_epsilon = 0.9
    alpha = 0.999
    n_iter = 10000
    win_rate = [0.25, 0.35, 0.45]
        
    eps_list = ["standard_epsilon", "decay_epsilon", "exp_epsilon"]
    
    optimal_idx = np.argmax(win_rate)
    for key  in eps_list:
        if key == 'standard_epsilon':
            epsilon = standard_epsilon(min_epsilon)
        bandit_list = [bandit(rate) for rate in win_rate]
        reward_list = []
        n_explore = 0
        n_exploit = 0
        n_optimal = 0
        for t in range(n_iter):
            if key == 'decaying_epsilon':
                epsilon = decaying_epsilon(min_epsilon, t)
            elif key == 'exp_epsilon':
                epsilon = exponential_decay_epsilon(min_epsilon, init_epsilon, alpha, t)
            
            if np.random.random() < epsilon:
                n_explore += 1
                idx = np.random.choice(len(bandit_list)) #Explore
            else:
                n_exploit += 1
                idx = np.argmax([bandit.sample_mean for bandit in bandit_list]) #Exploit
            
            if idx == optimal_idx:
                n_optimal += 1
                
            reward = int(bandit_list[idx].pull())
            reward_list.append(reward)
            bandit_list[idx].update(reward)
       
        for b in bandit_list:
            print(f"Under {key} : mean estimate is {b.sample_mean}")
        print("total reward earned:", np.sum(reward_list))
        print("overall win rate:", np.sum(reward_list) / n_iter)
        print("num_times_explored:", n_explore)
        print("num_times_exploited:", n_exploit)
        print("num times selected optimal bandit:", n_optimal)
        
        cumulative_rewards = np.cumsum(reward_list)
        win_rates = cumulative_rewards / (np.arange(n_iter) + 1)
        plt.plot(win_rates)
        plt.title(f"{key}")
        plt.show()

