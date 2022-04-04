import numpy as np
from scipy.optimize import fmin_tnc


class WrapperMNLRegression:

    def __init__(self, d, K, isBound=True):
        self.isBound = isBound
        if self.isBound:
            self.bounds = self._make_bounds(d, K)

    def _make_bounds(self, d, K):
        theta_bnds = [(-1., 1.) for _ in range(d)]
        log_lambda_bnds = [(0, np.exp(1)) for _ in range(K)]
        return theta_bnds + log_lambda_bnds

    def compute_prob(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means)
        u_ones = np.column_stack((u, np.ones(u.shape[0])))
        logSumExp = u_ones.sum(axis=1)
        prob = u_ones / logSumExp[:, None]
        return prob

    def cost_function(self, theta, *args):
        pass

    def gradient(self, theta, *args):
        pass

    def fit(self, theta, *args):
        if self.isBound:
            opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, bounds=self.bounds, args=args)
        else:
            opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=args)
        self.w = opt_weights[0]
        return self


class BasicMNLRegression(WrapperMNLRegression):

    def __init__(self, d, K, isBound=True):
        WrapperMNLRegression.__init__(self, d, K, isBound=isBound)

    def cost_function(self, theta, *args):
        x, y = args[0], args[1]
        m = x.shape[0]
        prob = self.compute_prob(theta, x, y)
        return -(1 / m) * np.sum(np.multiply(y, np.log1p(prob)))

    def gradient(self, theta, *args):
        x, y = args[0], args[1]
        m = x.shape[0]
        prob = self.compute_prob(theta, x, y)
        eps = (prob - y)[:, :-1]
        grad = (1 / m) * np.tensordot(eps, x, axes=([1, 0], [1, 0]))
        return grad


class RegularizedMNLRegression(WrapperMNLRegression):

    def __init__(self, d, K, isBound=True):
        WrapperMNLRegression.__init__(self, d, K, isBound=isBound)

    def cost_function(self, theta, *args):
        x, y, lam = args[0], args[1], args[2]
        m = x.shape[0]
        prob = self.compute_prob(theta, x, y)
        return -(1 / m) * np.sum(np.multiply(y, np.log(prob))) + (1 / m) * lam * np.linalg.norm(theta)

    def gradient(self, theta, *args):
        x, y, lam = args[0], args[1], args[2]
        m = x.shape[0]
        prob = self.compute_prob(theta, x, y)
        eps = (prob - y)[:, :-1]
        grad = (1 / m) * np.tensordot(eps, x, axes=([1, 0], [1, 0])) + (1 / m) * lam * theta
        return grad