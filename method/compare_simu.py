import numpy as np
from numba import jit
import numba

numba.config.NUMBA_NUM_THREADS = 4


# get the gradient of hat for the estimation of w's parameter matrix in method MWC
@jit(nopython=True, parallel=False)
def dev_probability(w, hat):
    pihat = 1 / (1 + np.exp(-hat))
    temp_loss = -np.mean(w * np.log(pihat) + (1 - w) * np.log(1 - pihat))
    dev = -w / (1 + np.exp(hat)) + (1 - w) / (1 + np.exp(-hat))
    return (dev, temp_loss)


# get the inverse probability weighted estimator's gradient, wei the weight, w the missing indicator, x the observed matrix, hat the x's parameter matrix's estimator
@jit(nopython=True, parallel=False)
def dev_weight_x(x, w, wei, hat):
    n1, n2 = x.shape
    dev = np.zeros((n1, n2))
    temp = np.zeros((n1, n2))
    dev = 2 * w * wei * (hat - x)
    temp = np.mean(w * wei * (x - hat) ** 2)
    return (dev, temp)


class matcom_compete:
    def __init__(self, x, w, k1, k2, threshold=10**-5):
        # use the compare estimators with known rank for x and w's parameter matrix as k2, k1
        np.random.seed(2024)
        n1, n2 = x.shape
        self.x = x
        self.w = w
        self.k1 = k1
        self.k2 = k2
        self.threshold = threshold
        self.n1 = n1
        self.n2 = n2
        self.up = np.random.randn(n1, k1)
        self.vp = np.random.randn(k1, n2)
        # the factor loadings and factors for w's parameter matrix
        self.u = np.random.randn(n1, k2)
        self.v = np.random.randn(k2, n2)
        # the factor loadings and factors for x's parameter matrix
        self.wei = np.ones((n1, n2))
        # the weight for estimation

    def loss(self):
        # compute the loss fucntion
        hat = self.u @ self.v
        temp_loss = np.mean((self.w * self.wei * (self.x - hat) ** 2))
        return temp_loss

    def update_onestep(self, eta=0.1):
        # update parameter for x with eta the learning rate
        # compute the gradient for x's estimation
        dev, loss = dev_weight_x(self.x, self.w, self.wei, self.u @ self.v)
        tempu = eta * (dev @ self.v.T / self.n2)
        tempv = eta * (self.u.T @ dev / self.n1)
        self.u -= tempu
        self.v -= tempv
        # update the factor loadings and factor for x
        return loss

    def update_onestep_pro(self, eta=0.1):
        # update the w's parameter with eta the learning rate
        dev, loss = dev_probability(self.w, self.up @ self.vp)
        tempup = eta * (dev @ self.vp.T / self.n2)
        tempvp = eta * (self.up.T @ dev / self.n1)
        self.up -= tempup
        self.vp -= tempvp
        return loss

    def estimate(self, eta=0.1, max_iter=1000000):
        # estimate x's parameter
        loss = np.array([])
        loss = np.append(loss, self.update_onestep(eta))
        loss = np.append(loss, self.update_onestep(eta))
        loss = np.append(loss, self.update_onestep(eta))
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.update_onestep(eta))
            i += 1
            # if i % 50 == 0:
            #     print(loss[i])
        self.loss = loss

    def estimate_pro(self, eta=0.1, max_iter=1000000):
        # estimate w's parameter and give the inverse probability weight for method MWC
        loss = np.array([])
        loss = np.append(loss, self.update_onestep_pro(eta))
        loss = np.append(loss, self.update_onestep_pro(eta))
        loss = np.append(loss, self.update_onestep_pro(eta))
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.update_onestep_pro(eta))
            i += 1
        hat = self.up @ self.vp
        self.wei = 1 + np.exp(-hat)
        self.wei = self.wei / np.mean(self.wei)
        self.loss_pro = loss

    def estimate_pro_NW(self):
        # estimate the inverse probability weight for method NW
        miss_row = np.mean(self.w, axis=1)
        miss_col = np.mean(self.w, axis=0)
        phat = miss_row.reshape(self.n1, 1) @ miss_col.reshape(1, self.n2)
        self.wei = 1 / phat
        self.wei /= np.mean(self.wei)

    def m1hat(self):
        return self.up @ self.vp

    def m2hat(self):
        return self.u @ self.v

    def mhat(self):
        return np.concatenate((self.m1hat(), self.m2hat()), axis=0)
