import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
import matplotlib.pyplot as plt
from multinomial import BasicMNLRegression

def sigmoid(means):
    return 1 / (1 + np.exp(-means))

class Env:
    def __init__(self, N, K, d, seed):
        self.N = N
        self.K = K
        self.d = d
        self.rand = np.random.RandomState(seed=seed)
        self.true_theta = np.array([1/2, 1/2, 1/2])

    def compute_reward(self, means):
        rwd = sigmoid(means)
        Y = np.array([self.rand.binomial(n=1, p=rwd[k]) for k in range(self.K)])
        return rwd, Y

    def get_optimal_reward(self, x):
        opt_means = np.sort(np.dot(x, self.true_theta))[::-1][:self.K]
        rwd, Y = self.compute_reward(opt_means)
        return rwd


class Firm:
    def __init__(self, timestep:int, fraudster: bool):
        self.fraudster = fraudster
        self.timestep = timestep
        self.caught = False
        self.active = True
        self.at, self.lt, self.ni = np.random.randn(3) + 2*int(fraudster)

    def decide_action(self, fraud: bool):
        self.at, self.lt, self.ni = np.random.randn(3) + 2*int(fraud)

    def update_active(self):
        if self.caught and self.fraudster:
            self.active = False

    def __repr__(self):
        return f"Firm active {self.active}, Fraudster {self.fraudster}, features {self.at, self.lt, self.ni}"

    def shift_fraud_distr(self, shift_mu):
        if self.fraudster:
            self.at, self.lt, self.ni = np.random.randn(3) + (2-shift_mu)

class FirmPool:
    "updates firm pool by extracting active firms & computing feature matrix with only active firms"
    def __init__(self, firms:list):
        self.pool = [firm for firm in firms if firm.active]
        self.N = N
        self.x = np.array([np.array([firm.at, firm.lt, firm.ni]) for firm in self.pool])

    def __repr__(self):
        return f"Pool length {len(self.pool)}"

class ConsUCB:
    """SEC uses ConsUCB to catch firms & update theta"""
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

    # def deactivate_firms(self, caught_firms: list):
    #     for firm in caught_firms:
    #         firm.caught = True
    #         firm.active = False




if __name__ == '__main__':
    firms = [Firm(0, False) for _ in range(100)] + [Firm(0, True) for _ in range(2)]
    N = len(firms)
    K = 2
    d = 3
    T = 100000

    tv = True # dataset is time-variant

    seed = 10
    alpha = 0.5
    model = 'cons-ucb'

    np.random.seed(seed)

    RWDS = []
    optRWD = []

    ########################## Environment #########################
    env = Env(N, K, d, seed)

    ########################## Model ###############################
    if model == 'cons-ucb':
        model = ConsUCB(N, K, d, seed=seed)
    else:
        raise Exception('Model should be "cons-ucb"')

    shift_times = 10
    shift_step = T / shift_times # every 10000 step, we shift fraudsters' distribution
    decrement_mu = 2 / shift_times # decrease mu by 0.002
    curr_decremented_mu = 0

    pool = firms

    feature_at = []
    feature_lt = []
    feature_ni = []

    for t in range(T):
        if t % shift_step == 0:
            curr_decremented_mu += decrement_mu
            for firm in pool:
                firm.shift_fraud_distr(curr_decremented_mu)
            print("fraud firm features:", pool[-1].at, pool[-1].lt, pool[-1].ni)
        print("Timestep:", t)
        catch_pool = FirmPool(pool)
        print(catch_pool)
        x = catch_pool.x

        S = model.choose_S(t, x)
        # print("chosen firms", S)
        rwd, Y = env.compute_reward(np.dot(x[S, :], env.true_theta))
        # print("Reward(Predicted prob of Committing Fraud)", rwd)
        # print("Y", Y)

        RWDS.append(rwd.sum())
        model.update_theta(x, Y, t+1)

        opt_rwd = env.get_optimal_reward(x)
        optRWD.append(opt_rwd.sum())

        index = S.tolist()
        for k, v in enumerate(Y.tolist()):
            if v == 1:
                remove_firm_index = S[k]
                remove_firm = pool[remove_firm_index]
                if remove_firm.fraudster:
                    pool.remove(pool[remove_firm_index])
                    to_insert = np.random.choice([i for i in range(len(pool))]).tolist()
                    print(to_insert)
                    pool.insert(to_insert, Firm(t, True))

    cumulated_regret = np.cumsum(optRWD) - np.cumsum(RWDS)
    plt.plot(np.arange(T), np.array(cumulated_regret))
    plt.legend()

    plt.ylabel('Total Regret')
    plt.xlabel('Time(t)')
    # plt.savefig(f'./image/total_regret_{title}.png')
    plt.show()

    plt.plot(np.array(feature_at))
    plt.xlabel('Time(t)')
    plt.ylabel('Gaussian Distribution Movement')
    plt.title('Feature Distribution movement of Fraudsters')
    plt.show()