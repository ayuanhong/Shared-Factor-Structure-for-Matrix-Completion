from pathlib import Path
import importlib.util

current_file = Path(__file__).resolve()
shfactor_path = current_file.parent / "shfactor.py"
spec = importlib.util.spec_from_file_location("shfactor", str(shfactor_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
globals().update({k: v for k, v in vars(module).items() if not k.startswith("_")})
temp_path = str(current_file.parent) + "/"

import numpy as np
from numba import jit
import numba

numba.config.NUMBA_NUM_THREADS = 4


# @jit(nopython = True,parallel = False)
def loss_mao(w, mhat1):
    # compute the loss function for w's parameter matrix of MWC
    n1, n2 = w.shape
    pihat = 1 / (1 + np.exp(-mhat1))
    temp_loss = -np.mean(w * np.log(pihat) + (1 - w) * np.log(1 - pihat))
    return temp_loss


@jit(nopython=True, parallel=False)
def lossd_mao(w, mhat1):
    # compute the gradient with loss fuction
    pihat = 1 / (1 + np.exp(-mhat1))
    temp_loss = -np.mean(w * np.log(pihat) + (1 - w) * np.log(1 - pihat))
    dev = -w / (1 + np.exp(mhat1)) + (1 - w) / (1 + np.exp(-mhat1))
    return (dev, temp_loss)


def onesteptruncate_trunc_mao(x, lam=1, k=10000):
    # the one step update for matrix x, with lam the parameter for nuclear penalty and k for truncated rank
    k = int(min(k, min(x.shape)))
    hatl, hats, hatr = trunsvd(x, k)
    temp_truncate = np.array([max(x - lam, 0) for x in hats])
    fullrank = np.sum(temp_truncate > 0)
    hatl = hatl[:, :fullrank]
    hatr = hatr[:fullrank, :]
    return [hatl, temp_truncate[:fullrank], hatr]


# @jit(nopython=True, parallel=False)
def onesteptruncate_mao(x, lam=1):
    # the one step update for matrix x, with lam the parameter for nuclear penalty and without truncated rank
    temp_svd = np.linalg.svd(x, full_matrices=False)
    temp_truncate = np.array([max(x - lam, 0) for x in temp_svd[1]])
    fullrank = np.sum(temp_truncate > 0)
    return (
        temp_svd[0][:, :fullrank],
        temp_truncate[:fullrank],
        temp_svd[2][:fullrank, :],
    )


@jit(nopython=True, parallel=False)
def onesteptruncate_mao_pre(x, lam=1):
    # the one step update for matrix x, with lam the parameter for nuclear penalty and without truncated rank for pre-training, as it doesn't use shrinkage for singular value
    # temp_svd = sp.linalg.svd(x, full_matrices=False)
    temp_svd = np.linalg.svd(x, full_matrices=False)
    temp_truncate = np.array([max(x - lam, 0) for x in temp_svd[1]])
    fullrank = np.sum(temp_truncate > 0)
    return (
        temp_svd[0][:, :fullrank],
        temp_truncate[:fullrank],
        temp_svd[2][:fullrank, :],
    )


class mao:
    # estimate the parameter matrix for w, the lam is the tuning parameter for nuclear penalty, L for the gradient proximal algorithm, k the truncated rank
    def __init__(self, w, k, lam, L=1, threshold=10**-5):
        np.random.seed(2024)
        n1, n2 = w.shape
        self.n1, self.n2 = w.shape
        self.w = w
        self.k = k
        self.lam = lam * np.log(min(n1, n2)) * np.sqrt(max(n1, n2))
        self.L = L
        self.threshold = threshold
        self.z = (
            np.random.randn(self.n1, self.k)
            @ np.random.randn(self.k, self.n2)
            / np.sqrt(self.k)
        )
        self.z -= np.mean(self.z)
        # need self.z the mean zero low-rank matrix
        self.mu = np.random.randn(1)
        # need self.mu the mean value for w's parameter matrix

    def loss_value(self):
        loss_value = loss_mao(self.w, self.z + self.mu)
        return loss_value

    def one_step(self):
        dev, loss = lossd_mao(self.w, self.z + self.mu)
        self.mu -= np.sum(dev) / (self.L * self.n1 * self.n2)
        u, s, v = onesteptruncate_mao(self.z - dev / self.L, self.lam / self.L)
        self.z = u @ np.diag(s) @ v
        self.mu += np.mean(self.z)
        self.z -= np.mean(self.z)
        return loss

    def one_step_pre(self):
        dev, loss = lossd_mao(self.w, self.z + self.mu)
        self.mu -= np.sum(dev) / (self.L * self.n1 * self.n2)
        u, s, v = onesteptruncate_mao_pre(self.z - dev / self.L, self.lam / self.L)
        self.z = u @ np.diag(s) @ v
        self.mu += np.mean(self.z)
        self.z -= np.mean(self.z)
        return loss

    def one_step_truncate(self):
        dev, loss = lossd_mao(self.w, self.z + self.mu)
        self.mu -= np.sum(dev) / (self.L * self.n1 * self.n2)
        u, s, v = onesteptruncate_trunc_mao(
            self.z - dev / self.L, self.lam / self.L, self.k
        )
        self.z = u @ np.diag(s) @ v
        self.mu += np.mean(self.z)
        self.z -= np.mean(self.z)
        return loss

    def estimate_pre(self, max_iter=5000):
        loss = np.array([])
        loss = np.append(loss, self.loss_value())
        loss = np.append(loss, self.one_step_pre())
        loss = np.append(loss, self.one_step_pre())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.one_step_pre())
            i += 1
        self.loss = loss

    def estimate(self, max_iter=5000):
        loss = np.array([])
        loss = np.append(loss, self.loss_value())
        loss = np.append(loss, self.one_step())
        loss = np.append(loss, self.one_step())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.one_step())
            i += 1
        self.loss = loss

    def estimate_truncate(self, iter=10, max_iter=5000):
        self.estimate_pre(iter)
        loss = np.array([])
        loss = np.append(loss, self.loss_value())
        loss = np.append(loss, self.one_step_truncate())
        loss = np.append(loss, self.one_step_truncate())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.one_step_truncate())
            i += 1
        self.loss_trunc = loss

    def rank(self):
        u, s, v = trunsvd(self.z + self.mu, self.k)
        return np.sum(s > 10 ** (-5))

    def hat(self):
        return self.z + self.mu

    def phat(self):
        esti = self.hat()
        return 1 / (1 + np.exp(-esti))

    def aic(self):
        r = self.rank()
        aic = self.loss_value() * self.n1 * self.n2 + r * (self.n1 + self.n2 - r)
        return [aic, r]


# similar for the compare_simu.py's functions
@jit(nopython=True, parallel=False)
def loss_wei(x, wei, mhat):
    return np.mean(wei * (x - mhat) ** 2 / 2)


@jit(nopython=True, parallel=False)
def lossd_wei(x, wei, mhat):
    loss = np.mean(wei * (x - mhat) ** 2)
    return (wei * (mhat - x), loss)


class mat_wei:
    # use nuclear penalty for inverse probability weighted methods, with k the truncated rank
    def __init__(self, x, wei, k, lam, L=1, threshold=10**-5):
        np.random.seed(2024)
        n1, n2 = x.shape
        self.n1, self.n2 = x.shape
        self.x = x
        self.wei = wei
        self.k = k
        self.lam = lam * np.log(min(n1, n2)) * np.sqrt(max(n1, n2))
        self.L = L
        self.threshold = threshold
        self.m = (
            np.random.randn(self.n1, self.k)
            @ np.random.randn(self.k, self.n2)
            / np.sqrt(self.k)
        )

    def lossd(self):
        return lossd_wei(self.x, self.wei, self.m)

    def loss_value(self):
        dev, loss = self.lossd()
        return loss

    def one_step(self):
        dev, loss = self.lossd()
        temp = self.m - dev / self.L
        u, s, v = onesteptruncate_mao(temp, self.lam / self.L)
        self.m = u @ np.diag(s) @ v
        return loss

    def one_step_pre(self):
        dev, loss = self.lossd()
        temp = self.m - dev / self.L
        u, s, v = onesteptruncate_mao_pre(temp, self.lam / self.L)
        self.m = u @ np.diag(s) @ v
        return loss

    def one_step_truncate(self):
        dev, loss = self.lossd()
        temp = self.m - dev / self.L
        u, s, v = onesteptruncate_trunc_mao(temp, self.lam / self.L, self.k)
        self.m = u @ np.diag(s) @ v
        return loss

    def estimate(self, max_iter=5000):
        loss = np.array([])
        loss = np.append(loss, self.loss_value())
        loss = np.append(loss, self.one_step())
        loss = np.append(loss, self.one_step())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.one_step())
            i += 1
        self.loss = loss

    def estimate_pre(self, max_iter=5000):
        loss = np.array([])
        loss = np.append(loss, self.loss_value())
        loss = np.append(loss, self.one_step_pre())
        loss = np.append(loss, self.one_step_pre())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.one_step_pre())
            i += 1
        self.loss = loss

    def estimate_truncate(self, iter=10, max_iter=5000):
        self.estimate(iter)
        loss = np.array([])
        loss = np.append(loss, self.loss_value())
        loss = np.append(loss, self.one_step_truncate())
        loss = np.append(loss, self.one_step_truncate())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.one_step_truncate())
            i += 1
        self.loss_trunc = loss

    def refresh(self):
        self.m = (
            np.random.randn(self.n1, self.k)
            @ np.random.randn(self.k, self.n2)
            / np.sqrt(self.k)
        )

    def rank(self):
        u, s, v = trunsvd(self.m, self.k)
        return np.sum(s > 10 ** (-5))

    def cv(self, fold=5):
        # use cv for fold-5
        np.random.seed(2024)
        wei_save = self.wei
        w = self.wei > 10 ** (-5)
        w = w + 0
        temp = w.reshape(-1)
        index = np.where(temp > 0.9)
        index = np.array(index).reshape(-1)
        kf = KFold(fold, shuffle=True)
        cvset = list()
        mse = np.zeros(fold)
        rank = np.zeros(fold)
        for i in kf.split(index):
            temp_ = np.zeros(self.n1 * self.n2)
            temp_[index[i[1]]] = 1
            cvset.append(temp_.reshape(self.n1, self.n2))
        for i in range(fold):
            train = w * (1 - cvset[i])
            test = cvset[i]
            self.wei = train * wei_save
            self.estimate(10)
            self.estimate_truncate()
            mse[i] = np.sum(test * (self.x - self.m) ** 2) / np.sum(test)
            rank[i] = self.rank()
            self.refresh()
            self.wei = wei_save
        return [np.mean(mse), np.mean(rank)]
