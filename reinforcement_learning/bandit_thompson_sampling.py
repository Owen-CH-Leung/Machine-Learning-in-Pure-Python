import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class bandit:
    def __init__(self, mean_rate):
        self.mean_rate = mean_rate
        self.m = 0
        self.lambda_ = 1
        self.sum_x = 0 # for convenience
        self.tau = 1
        self.N = 0
        
    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.mean_rate

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m
    
    def update(self, reward):
        self.lambda_ += self.tau
        self.sum_x += reward
        self.m = self.tau*self.sum_x / self.lambda_
        self.N += 1

def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for b in bandits:
      y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
      plt.plot(x, y, label=f"real mean: {b.mean_rate:.4f}, num plays: {b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n_iter = 10000
    mean_rate = [1.25, 2.35, 3.45]
    plot_pts = [5,100,1000,5000,9999]
    optimal_idx = np.argmax(mean_rate)
    bandit_list = [bandit(rate) for rate in mean_rate]
    reward_list = []
    n_optimal = 0
    N = 0
    for t in range(n_iter):
        if t in plot_pts:
            plot(bandit_list, t)
        N += 1
        idx = np.argmax([b.sample() for b in bandit_list]) #Thompson Sampling
        if idx == optimal_idx:
            n_optimal += 1
        reward = int(bandit_list[idx].pull())
        reward_list.append(reward)
        bandit_list[idx].update(reward)
   
    for b in bandit_list:
        print(f"mean estimate is {b.m}")
    print("total reward earned:", np.mean(reward_list))
    print("overall win rate:", np.mean(reward_list))
    print("num times selected optimal bandit:", n_optimal)
    