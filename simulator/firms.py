import numpy as np
from abc import ABCMeta, abstractmethod
from pytz import timezone, country_timezones
from mesa import Agent, Model
import pandas as pd
from tqdm import tqdm
from consucb import ConsUCB, SimulatedDataset
from numpy.linalg import inv

def data_extract(fraudster, data=r'./../uscecchini_manip.csv'):
    value = int(fraudster) # 1 if fraudster=True, 0 if fraudster=False
    df = pd.read_csv(data, index_col=False)
    df1 = df.loc[df['misstate']==value]
    return df1

def sigmoid(means):
    return 1 / (1 + np.exp(-means))

class Env:
    """
    creates the environment, takes all firms and selects only those firms that are ACTIVE
    excludes those firms that are inactive (caught by SEC) and provides the pool (list of firms) to SEC
    """
    def __init__(self, N, K, d, seed):
        self.N = N # number of firms
        self.K = K # K arms
        self.d = d # size

        self.rand = np.random.RandomState(seed=seed)

        self.theta = self.rand.uniform(0., 1., self.d) * 2 - 1  # True Theta (-1 ~ +1 uniform)

    def create_pool(self, pool):
        """
        :param pool: list of all firm (list of classes; Firm class)
        :return:
            catch_pool : list of firms eligible to be inspected (active firms)
            X : 2d numpy array of firm features
            Y : binary integer (0/1); 1 if firm is a fraudster, else 0
        """
        catch_pool = []
        array = []
        Y = []
        for p in pool:
            if p.active:
                catch_pool.append(p)
                array.append([p.at, p.cogs, p.lt, p.ni, p.ch_roa, p.ebit, p.ch_fcf])
                Y.append(int(p.fraudster))
        X = np.array(array)
        Y = np.array(Y)
        return catch_pool, X, Y


    def compute_reward(self, means):
        rwd = sigmoid(means)
        Y = np.array([self.rand.binomial(n=1, p=rwd[k]) for k in range(self.K)])
        return rwd, Y

    def get_optimal_reward(self, x):
        opt_means = np.sort(np.dot(x, self.theta))[::-1][:self.K]
        rwd, Y = self.compute_reward(opt_means)
        return rwd

class Firm:

    def __init__(self, timestep, fraudster:bool):
        self.fraudster = fraudster
        self.timestep = timestep
        self.caught = None
        self.active = True

        df = data_extract(self.fraudster)
        self.d = 7 # number of features
        # feature extraction for each firm
        self.at = np.std(df['at'])*np.random.randn() + np.mean(df['at'])
        self.cogs = np.std(df['cogs'])*np.random.randn() + np.mean(df['cogs'])
        self.lt = np.std(df['lt'])*np.random.randn() + np.mean(df['lt'])
        self.ni = np.std(df['ni'])*np.random.randn() + np.mean(df['ni'])
        self.ch_roa = np.std(df['ch_roa'])*np.random.randn() + np.mean(df['ch_roa'])
        self.ebit = np.std(df['ebit'])*np.random.randn() + np.mean(df['ebit'])
        self.ch_fcf = np.std(df['ch_fcf'])*np.random.randn() + np.mean(df['ch_fcf'])

    def update_active(self):
        if self.caught:
            if self.fraudster:
               self.active = False

    def decide_fraud(self, stay_genuine_prob=0.99, stay_fraud_prob=0.9):
        pass

    def compute_reward(self):
        pass

    def __repr__(self):
        return f"Firm active status {self.active}, Fraudster {self.fraudster}, firm features {self.at, self.cogs, self.lt, self.ni, self.ch_roa, self.ebit, self.ch_fcf}"

class SEC:
    def __init__(self, N, K=2, d=7, seed=10, alpha=0.5):
        self.N = N
        self.K = K #arms to pull
        self.d = d # number of features
        self.timestep = 0
        self.seed = seed
        self.alpha = alpha

        #UCB model implementation for catching firms
        self.theta = np.zeros(self.d)
        self.A = np.eye(self.d)
        self.b = np.zeros(self.d)

    def catch_Sfirms(self, x):
        A_k = self.A
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
        # print(x[self.S, :].shape, Y.shape)
        self.A += np.matmul(x.T, x)
        self.b += np.matmul(x.T, Y)
        self.theta = inv(self.A) @ self.b



if __name__ == '__main__':
    firms = [Firm(0, False) for _ in range(10)] + [Firm(0, True) for _ in range(2)]
    caught_firms = []
    T = 200
    seed = 123
    sec = SEC(len(firms))
    env = Env(len(firms), sec.K, sec.d, seed=seed)
    for t in range(1, T):
        sec.timestep = t
        catch_pool, X, Y = env.create_pool(firms)
        print(X.shape, Y.shape)
        sec.N = len(catch_pool)
        caught_firm_index = sec.catch_Sfirms(X)
        caught_firms.append(caught_firm_index)
        print(f"caught firms at time {t} is {caught_firms}")
        sec.update_theta(X, Y, t)
        print(f"updated theta is {sec.theta}")
        for i in caught_firm_index:
            p = firms[i]
            p.active=False
            p.caught=True
            p.update_active()
            print(f"index {i} firm has been updated to false")

        catch_pool.extend([Firm(t, True) for _ in range(len(caught_firm_index))])
        print("length of catch pool:", len(catch_pool))