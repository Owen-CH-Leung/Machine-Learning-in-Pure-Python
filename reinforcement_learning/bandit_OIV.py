import numpy as np
import matplotlib.pyplot as plt

class bandit:
    def __init__(self, win_rate):
        self.win_rate = win_rate
        self.n = 1
        self.sample_mean = 5 #Optimistic Initial Value
        
    def pull(self):
        return np.random.random() < self.win_rate
        
    def update(self, reward):
        self.n += 1
        learning_rate = 1 / self.n
        self.sample_mean = self.sample_mean + learning_rate * (reward - self.sample_mean)

if __name__ == '__main__':
    n_iter = 10000
    win_rate = [0.25, 0.35, 0.45]

    
    optimal_idx = np.argmax(win_rate)
    bandit_list = [bandit(rate) for rate in win_rate]
    reward_list = []
    n_optimal = 0
    for t in range(n_iter):
        idx = np.argmax([bandit.sample_mean for bandit in bandit_list]) #Always Exploit, no explore
        if idx == optimal_idx:
            n_optimal += 1
        reward = int(bandit_list[idx].pull())
        reward_list.append(reward)
        bandit_list[idx].update(reward)
   
    for b in bandit_list:
        print(f"mean estimate is {b.sample_mean}")
    print("total reward earned:", np.sum(reward_list))
    print("overall win rate:", np.sum(reward_list) / n_iter)
    print("num times selected optimal bandit:", n_optimal)
    
    cumulative_rewards = np.cumsum(reward_list)
    win_rates = cumulative_rewards / (np.arange(n_iter) + 1)
    plt.plot(win_rates)
    plt.show()