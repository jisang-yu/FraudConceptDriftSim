import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import inv
import matplotlib.pyplot as plt

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

    def fraud_distr_shift(self, step, diff = 10, features=['at', 'cogs', 'lt', 'ni', 'ch_roa', 'ebit', 'ch_fcf']):
        # divide the difference into 10 steps, and then compute the current step
        df_gen = data_extract(False)[features]
        df_frd = data_extract(True)[features]
        fstd = df_frd.std()
        diff = (df_gen.mean() - df_frd.mean()) / diff
        new_frd = df_frd.mean() + diff * step
        df = pd.concat([new_frd, fstd], axis=1)

        # draw from new feature distribution; column 0 : mean / column 1 : stddev
        self.at = df[1]['at']*np.random.randn() + df[0]['at']
        self.cogs = df[1]['cogs'] * np.random.randn() + df[0]['cogs']
        self.lt = df[1]['lt'] * np.random.randn() + df[0]['lt']
        self.ni = df[1]['ni'] * np.random.randn() + df[0]['ni']
        self.ch_roa = df[1]['ch_roa'] * np.random.randn() + df[0]['ch_roa']
        self.ebit = df[1]['ebit'] * np.random.randn() + df[0]['ebit']
        self.ch_fcf = df[1]['ch_fcf'] * np.random.randn() + df[0]['ch_fcf']

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
        # X: 12*7 / Y 12*1 / self.A 7*7, self.b 7*1 /
        # idx = np.array(self.S)
        # print(x[idx], x[idx].shape)
        self.A += np.matmul(x[self.S, :].T, x[self.S, :])
        self.b += np.matmul(x[self.S, :].T, Y[self.S])
        self.theta = inv(self.A) @ self.b



if __name__ == '__main__':
    firms = [Firm(0, False) for _ in range(100)] + [Firm(0, True) for _ in range(2)]
    print(len(firms))
    T = 100000
    distr_shift_num = 10
    distr_shift_step = T / distr_shift_num #every 40 steps we change the distribution of fraudster
    seed = 123
    sec = SEC(len(firms))
    env = Env(len(firms), sec.K, sec.d, seed=seed)
    # firm_size = []
    caught_counts_array = []
    caught_firm_features = []

    for t in range(1, T):
        inspected_firms = []
        sec.timestep = t
        print(sec.timestep)
        catch_pool, X, Y = env.create_pool(firms)
        print(len(catch_pool))
        print(Y)
        sec.N = len(catch_pool)
        inspected_firm_index = sec.catch_Sfirms(X)
        inspected_firms.append(inspected_firm_index)
        print(f"Inspected firms at time {t} is {inspected_firms}")
        sec.update_theta(X, Y, t)
        print(f"updated theta is {sec.theta}")
        caught_count = 0
        for i in inspected_firm_index:
            if Y[i] == 1:
                caught_count += 1
                p = firms[i]
                caught_firm_features.append([p.at, p.cogs, p.lt, p.ni, p.ch_roa, p.ebit, p.ch_fcf])
                p.active=False
                p.caught=True
                p.update_active()
                print("Firm is caught and status changed to :", p, p.active, p.caught)

        new_fraudsters = [Firm(t, True) for _ in range(caught_count)]
        step = t / distr_shift_step
        if step >= 1 and new_fraudsters:
            print(new_fraudsters)
            for f in new_fraudsters:
                f.fraud_distr_shift(step=step)
            f = new_fraudsters[0]
            print("new fraudster features\n", f.at, f.cogs, f.lt, f.ni, f.ch_roa, f.ebit, f.ch_fcf)
        catch_pool.extend(new_fraudsters)
        firms = catch_pool
        # print("length of catch pool:", len(firms))
        # firm_size.append(len(firms))
        caught_counts_array.append(caught_count)
        print(f"history of counts of caught fraudsters' at {sec.timestep} is {caught_counts_array}")

    #plot firm sizes

    plt.plot(np.arange(T-1), np.array(caught_counts_array))
    plt.ylabel("Firm Sizes")
    plt.xlabel("Time(t)")
    plt.title("Firm Size Growth, more means firms caught every timestep")
    plt.show()


    df = data_extract(True)
    a, b, c, d, e, f, g  = df['at'].mean(), \
                       df['cogs'].mean(), \
                       df['lt'].mean(), \
                       df['ni'].mean(), \
                       df['ch_roa'].mean(), \
                       df['ebit'].mean(), \
                       df['ch_fcf'].mean()
    print(a, b, c, d, e, f, g)
    map = {1: ['at', a], 2: ['cogs', b], 3: ['lt', c], 4: ['ni', d], 5: ['ch_roa', e], 6: ['ebit', f], 7: ['ch_fcf', g]}

    for n in range(7):
        print(caught_firm_features)
        # x = np.linspace(0, len(caught_firm_features))
        avg = map[n+1][1]
        plt.plot(np.array([i[n] for i in caught_firm_features]))
        plt.plot(avg, 'go--', linestyle='dashed')
        plt.ylabel(f"{map[n+1]} of caught-firms")
        plt.xlabel("Timestep (t)")
        plt.title(f"{map[n+1][0]} trend of caught-firm, and fraudster average (distribution drawn from)")
        plt.show()
