import numpy as np
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv


############# Environment ################

def sigmoid(means):
    return 1 / (1 + np.exp(-means))


class Env:
    def __init__(self, N, K, d, seed):
        self.N = N # number of firms
        self.K = K # K arms
        self.d = d # size

        self.rand = np.random.RandomState(seed=seed)

        self.theta = self.rand.uniform(0., 1., self.d) * 2 - 1  # True Theta (-1 ~ +1 uniform)

    def compute_reward(self, means):
        rwd = sigmoid(means)
        Y = np.array([self.rand.binomial(n=1, p=rwd[k]) for k in range(self.K)])
        return rwd, Y

    def get_optimal_reward(self, x):
        opt_means = np.sort(np.dot(x, self.theta))[::-1][:self.K]
        rwd, Y = self.compute_reward(opt_means)
        return rwd


############# Dataset ################

class SimulatedDataset:

    def __init__(self, N, K, d, seed):
        self.N = N
        self.K = K
        self.d = d

        self.seed = seed
        self.rand = np.random.RandomState(seed=seed)

        self.update_simulated_data()

    def update_simulated_data(self):
        self.x = (self.rand.random((self.N, self.d)) * 2 - 1)


############# Model ################

class ConsUCB:
    def __init__(self, N, K, d, seed, alpha=0.5):
        self.N = N
        self.K = K
        self.d = d

        self.seed = seed
        self.rand = np.random.RandomState(seed=seed)

        self.alpha = alpha

        self.theta = np.zeros(self.d)

        self.A = np.eye(self.d)
        self.b = np.zeros(self.d)

    def choose_S(self, t, x):  # x is N*d matrix

        A_k = self.A  # Initialize A_t,0 = A_t-1

        S = []

        for k in range(self.K):
            means = np.dot(x, self.theta)
            xA = np.sqrt((np.matmul(x, inv(self.A)) * x).sum(axis=1))
            xAk = np.sqrt((np.matmul(x, inv(A_k)) * x).sum(axis=1))

            p = means - self.alpha * xA + 2 * self.alpha * xAk

            p[S] = -np.inf

            chosen_idx = np.argsort(p)[::-1][0]

            S.append(chosen_idx)
            A_k = A_k + (x[chosen_idx].T @ x[chosen_idx])

        self.S = np.array(S)
        print(self.S)

        return (self.S)

    def update_theta(self, x, Y, t):
        self.A += np.matmul(x[self.S, :].T, x[self.S, :])
        self.b += np.matmul(x[self.S, :].T, Y)

        self.theta = inv(self.A) @ self.b


def main():
    ############# Parameter ################

    N = 1000  # the number of Firms
    K = 10  # the number of chosen firms
    d = 100  # feature dimensions
    T = 5000  # time periods

    tv = False  # if tv= True, then the dataset will be time-variant

    seed = 10
    alpha = 0.5
    model = 'cons-ucb'

    random.seed(seed)


    RWDS = []
    optRWD = []

    ########################## Environment #########################
    env = Env(N, K, d, seed)

    ########################## Model ###############################
    if model == 'cons-ucb':
        model = ConsUCB(N, K, d, seed=seed)
    else:
        raise Exception('Model should be "cons-ucb"')

    ######################### Data #################################
    dataset = SimulatedDataset(N, K, d, seed)

    ########################## Algorithm ############################
    start = time.time()
    for t in range(T):
        if tv and t > 0:
            dataset.update_simulated_data()

        x = dataset.x

        S = model.choose_S(t + 1, x)
        rwd, Y = env.compute_reward(np.dot(x[S, :], env.theta))

        RWDS.append(rwd.sum())
        model.update_theta(x, Y, t + 1)

        opt_rwd = env.get_optimal_reward(x)
        optRWD.append(opt_rwd.sum())

    end = time.time()
    runtime = np.array([end - start])

    cumulated_regret = np.cumsum(optRWD) - np.cumsum(RWDS)

    ########################## Plot  #################################

    # title = 'title'

    plt.plot(np.arange(T), np.array(cumulated_regret))
    plt.legend()

    plt.ylabel('Total Regret')
    plt.xlabel('Time(t)')
    # plt.savefig(f'./image/total_regret_{title}.png')
    plt.show()

if __name__ == '__main__':
    main()