import numpy as np
from numba import jit, vectorize, prange
import numba

numba.config.NUMBA_NUM_THREADS = 4
import sklearn as sk

# import numba_scipy


# define the cap function
@jit(nopython=True, parallel=False)
def cap(x, lam, gam):
    return lam * min(1, x / gam)


# define mcp function
@jit(nopython=True, parallel=False)
def mcp(x, lam, gam):
    if 0 <= x <= lam * gam:
        return lam * x - x**2 / (2 * gam)
    else:
        return lam**2 * gam / 2


# define scad function
@jit(nopython=True, parallel=False)
def scad(x, lam, gam):
    if 0 <= x <= lam:
        return lam * x
    elif x <= lam * gam:
        return lam * x - (x - lam) ** 2 / (2 * gam - 2)
    else:
        return lam**2 * (gam + 1) / 2


# define nuclear function
@jit(nopython=True, parallel=False)
def nuclear(x, lam, gam):
    return lam * x


# the singular value threshold function for cap loss
@jit(nopython=True, parallel=False)
def scap(x, lam, gam, tau):
    if 0 < lam < 2 * tau * gam**2:
        if x <= lam / gam / tau:
            return 0
        elif x < gam + lam / 2 / gam / tau:
            return x - lam / gam / tau
        else:
            return x
    else:
        if x < np.sqrt(2 * lam / tau):
            return 0
        else:
            return x


# the singular value threshold function for mcp loss
@jit(nopython=True, parallel=False)
def smcp(x, lam, gam, tau):
    if gam * tau > 1:
        if x <= lam / tau:
            return 0
        elif x <= gam * lam:
            return gam * (tau * x - lam) / (gam * tau - 1)
        else:
            return x
    else:
        if x < np.sqrt(gam / tau) * lam:
            return 0
        else:
            return x


# the singular value threshold function for scad loss
@jit(nopython=True, parallel=False)
def sscad(x, lam, gam, tau):
    if tau > 1 / (gam - 1):
        if x <= lam / tau:
            return 0
        elif x <= (lam / tau + lam):
            return x - lam / tau
        elif x <= gam * lam:
            return ((gam - 1) * tau * x - gam * lam) / ((gam - 1) * tau - 1)
        else:
            return x
    elif tau <= 1 / (gam - 1):
        if x <= lam / tau:
            return 0
        elif x <= (lam / 2 / tau + (gam + 1) * lam / 2):
            return x - lam / tau
        else:
            return x
    else:
        if x <= np.sqrt((gam + 1) / tau) * lam:
            return 0
        else:
            return x


# the singular value threshold function for nuclear loss
@jit(nopython=True, parallel=False)
def snuclear(x, lam, gam, tau):
    return max(x - lam / tau, 0)


# the truncated SVD for matrix x, with k the truncated rank
def trunsvd(x, k=10000):
    k = int(min(k, min(x.shape)))
    k = max(k, 1)
    temp_truncate = sk.decomposition.TruncatedSVD(n_components=k).fit(x)
    hats = temp_truncate.singular_values_
    hatr = temp_truncate.components_
    hatl = x @ hatr.T @ np.diag(1 / hats)
    return [hatl, hats, hatr]


# the one step update for matrix x, with k the truncated rank, lam the tuning parameter of loss function, gam the scale parameter, and tau used for gradient proximal algorithm
def onesteptruncate_trunc(x, fun, k=10000, lam=1, gam=1.5, tau=1):
    k = int(min(k, min(x.shape)))
    hatl, hats, hatr = trunsvd(x, k)
    temp_truncate = np.array([fun(x, lam, gam, tau) for x in hats])
    fullrank = np.sum(temp_truncate > 0)
    hatl = hatl[:, :fullrank]
    hatr = hatr[:fullrank, :]
    return [hatl, temp_truncate[:fullrank], hatr]


# the update for matrix x without truncated rank
@jit(nopython=True, parallel=False)
def onesteptruncate(x, fun, lam, gam, tau):
    temp_svd = np.linalg.svd(x, full_matrices=False)
    temp_truncate = [fun(y, lam, gam, tau) for y in temp_svd[1]]
    temp_truncate = np.array(temp_truncate)
    fullrank = np.sum(temp_truncate > 0)
    return (
        temp_svd[0][:, :fullrank],
        temp_truncate[:fullrank],
        temp_svd[2][:fullrank, :],
    )


# the loss function for parameter mhat
@jit(nopython=True, parallel=False)
def loss(x, w, mhat1, mhat2, eta=1):
    n1, n2 = x.shape
    temp_loss = -np.mean(w * mhat1) + np.mean(np.log(1 + np.exp(mhat1)))
    temp = np.zeros((n1, n2))
    for i in prange(n1):
        for j in prange(n2):
            if w[i, j] > 0:
                temp[i, j] = 1 / 2 * (x[i, j] - mhat2[i, j]) ** 2
    temp_loss += np.mean(temp) * eta
    return temp_loss


# the loss function with its gradient for parameter mhat
@jit(nopython=True, parallel=False)
def lossd(x, w, mhat1, mhat2, eta=1):
    n1, n2 = x.shape
    dev = np.zeros((2 * n1, n2))
    temp_loss = -np.mean(w * mhat1) + np.mean(np.log(1 + np.exp(mhat1)))
    temp = np.zeros((n1, n2))
    temp_dev_2 = np.zeros((n1, n2))
    for i in prange(n1):
        for j in prange(n2):
            if w[i, j] > 0:
                temp[i, j] = w[i, j] * 1 / 2 * (x[i, j] - mhat2[i, j]) ** 2
                temp_dev_2[i, j] = mhat2[i, j] - x[i, j]
    temp_loss += np.mean(temp) * eta
    dev[:n1, :] = -w + 1 / (1 + np.exp(-mhat1))
    dev[n1:, :] = temp_dev_2 * eta
    return (dev, temp_loss)


# @jit(nopython = True,parallel = False)
# get the factor and loadings with matrix m1hat, m2hat, which has k shared factor and k1, k2 specific factor
def get_factor_loadings(m1hat, m2hat, k, k1, k2):
    if (k + k1 + k2) == 0:
        k = 1
    d = k + k1 + k2
    n1, n2 = m1hat.shape
    temp1 = trunsvd(m1hat, d)
    temp2 = trunsvd(m2hat, d)
    v1 = temp1[2][: (k + k1), :]
    v2 = temp2[2][: (k + k2), :]
    m1hat = temp1[0][:, : (k + k1)] @ np.diag(temp1[1][: (k + k1)]) @ v1
    m2hat = temp2[0][:, : (k + k2)] @ np.diag(temp2[1][: (k + k2)]) @ v2
    m1hat_co = m1hat @ v2.T @ v2
    m1hat_in = m1hat - m1hat_co
    m2hat_co = m2hat @ v1.T @ v1
    m2hat_in = m2hat - m2hat_co
    mhat_co = np.concatenate((m1hat_co, m2hat_co), axis=0)
    temp_co = trunsvd(mhat_co, d)
    temp1 = trunsvd(m1hat_in, d)
    temp2 = trunsvd(m2hat_in, d)
    u_co = temp_co[0][:, :(k)] @ np.diag(np.sqrt(temp_co[1][:(k)]))
    # the shared factor loading part
    v_co = np.diag(np.sqrt(temp_co[1][:(k)])) @ temp_co[2][:(k), :]
    # the shared factor part
    u_in1 = temp1[0][:, :k1] @ np.diag(np.sqrt(temp1[1][:k1]))
    # the specific factor loading part for mhat1
    v_in1 = np.diag(np.sqrt(temp1[1][:k1])) @ temp1[2][:k1, :]
    # corresponding specific factor for mhat1
    u_in2 = temp2[0][:, :k2] @ np.diag(np.sqrt(temp2[1][:k2]))
    v_in2 = np.diag(np.sqrt(temp2[1][:k2])) @ temp2[2][:k2, :]
    u1 = np.concatenate((u_in1, u_co[:n1, :]), axis=1)
    u2 = np.concatenate((u_co[n1:, :], u_in2), axis=1)
    v = np.concatenate((v_in1, v_co, v_in2), axis=0)
    # u1 as the k + k1 factor loading for mhat1, with k1:k1+k the shared part
    # u2 the k + k2 factor loading for mhat2, with 0:k the shared part
    # v the collected factors
    return [u1, u2, v]


@vectorize
def truncate(x, a, b):
    # truncate matrix x
    if x <= a:
        if x >= b:
            return x
        else:
            return b
    else:
        return a


# the first step estimation that use the matrix loss function for regularization
class matcom:
    def __init__(self, x, w, k, penalty, lam, gam, mu, eta=1, threshold=10**-5):
        # x the observed matrix, w the missing indicator, k the preselect rank, penalty the selected penalty function, lam, gam the parameter for penalty, mu choose for proximal gradient algorithm, eta the weight part between x and w, threshold control the stopping.
        np.random.seed(2024)
        n1, n2 = x.shape
        self.x = x
        self.w = w
        self.k = k
        self.lam = lam * np.log(min(n1, n2)) * np.sqrt(max(n1, n2))
        # use the scale factor for tuning parameter lam
        self.gam = gam
        self.mu = mu
        self.threshold = threshold
        self.eta = eta
        self.n1 = n1
        self.n2 = n2
        self.u = np.random.randn(2 * n1, k) / np.power(k, 1 / 4)
        self.s = np.ones(k)
        self.v = np.random.randn(k, n2) / np.power(k, 1 / 4)
        match penalty:
            case "nuclear":
                self.penalty = nuclear
                self.penalty_update = snuclear
            case "cap":
                self.penalty = cap
                self.penalty_update = scap
            case "mcp":
                self.penalty = mcp
                self.penalty_update = smcp
            case "scad":
                self.penalty = scad
                self.penalty_update = sscad
            case _:
                self.penalty = nuclear
                self.penalty_update = snuclear

    def change_penalty(self, penalty):
        # change the penalty function
        match penalty:
            case "nuclear":
                self.penalty = nuclear
                self.penalty_update = snuclear
            case "cap":
                self.penalty = cap
                self.penalty_update = scap
            case "mcp":
                self.penalty = mcp
                self.penalty_update = smcp
            case "scad":
                self.penalty = scad
                self.penalty_update = sscad
            case _:
                self.penalty = nuclear
                self.penalty_update = snuclear

    def loss(self):
        # compute the loss fuction
        hat1 = self.u[: self.n1, :] @ np.diag(self.s) @ self.v
        hat2 = self.u[self.n1 :, :] @ np.diag(self.s) @ self.v
        loss_temp = loss(self.x, self.w, hat1, hat2, self.eta)
        return loss_temp

    def dev(self):
        # compute the loss fuction and the gradient
        hat = self.u @ np.diag(self.s) @ self.v
        dev, loss_temp = lossd(
            self.x, self.w, hat[: self.n1, :], hat[self.n1 :, :], self.eta
        )
        return dev

    def update_onestep(self):
        # use one step update without rank truncation
        hat = self.u @ np.diag(self.s) @ self.v
        dev, loss_temp = lossd(
            self.x, self.w, hat[: self.n1, :], hat[self.n1 :, :], self.eta
        )
        loss_temp += np.sum([self.penalty(x, self.lam, self.gam) for x in self.s]) / (
            self.n1 * self.n2
        )
        temp_y = hat - 1 / self.mu * dev
        self.u, self.s, self.v = onesteptruncate(
            temp_y, self.penalty_update, self.lam, self.gam, self.mu
        )
        return loss_temp

    def update_onestep_truncate(self):
        # use self.k truncated update
        hat = self.u @ np.diag(self.s) @ self.v
        dev, loss_temp = lossd(
            self.x, self.w, hat[: self.n1, :], hat[self.n1 :, :], self.eta
        )
        loss_temp += np.sum([self.penalty(x, self.lam, self.gam) for x in self.s]) / (
            self.n1 * self.n2
        )
        temp_y = hat - 1 / self.mu * dev
        self.u, self.s, self.v = onesteptruncate_trunc(
            temp_y, self.penalty_update, self.k, self.lam, self.gam, self.mu
        )
        return loss_temp

    def update_onestep_pre(self):
        # use the pre training befor the matrix mcp
        hat = self.u @ np.diag(self.s) @ self.v
        dev, loss_temp = lossd(
            self.x, self.w, hat[: self.n1, :], hat[self.n1 :, :], self.eta
        )
        temp_y = hat - 1 / self.mu * dev
        u, s, v = trunsvd(temp_y, self.k)
        fullrank = np.sum(s > self.lam)
        self.u = u[:, :fullrank]
        self.s = s[:fullrank]
        self.v = v[:fullrank, :]
        return loss_temp

    def estimate(self, max_iter=10000):
        # use the estimation without rank truncate
        loss = np.array([])
        loss = np.append(loss, self.loss())
        loss = np.append(loss, self.update_onestep())
        loss = np.append(loss, self.update_onestep())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.update_onestep())
            i += 1
            if self.s.shape[0] == 0:
                break
        self.loss_trun = loss

    def estimate_trunc(self, max_iter=10000):
        # use the estimation with rank truncate
        loss = np.array([])
        loss = np.append(loss, self.loss())
        loss = np.append(loss, self.update_onestep_truncate())
        loss = np.append(loss, self.update_onestep_truncate())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.update_onestep_truncate())
            if self.s.shape[0] == 0:
                break
            i += 1
        if i < 50:
            # at least update 50 iterations
            for i in range(i, 50, 1):
                loss = np.append(loss, self.update_onestep_truncate())
                if self.s.shape[0] == 0:
                    break
        self.loss_trun = loss

    def estimate_pre(self, max_iter=10000):
        # similalry use the update for pre training
        loss = np.array([])
        loss = np.append(loss, self.loss())
        loss = np.append(loss, self.update_onestep_pre())
        loss = np.append(loss, self.update_onestep_pre())
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.update_onestep_pre())
            i += 1
        self.loss_pre = loss

    def m1hat(self):
        # return the estimator for mhat1
        return (self.u @ np.diag(self.s) @ self.v)[: self.n1, :]

    def m2hat(self):
        return (self.u @ np.diag(self.s) @ self.v)[self.n1 :, :]

    def mhat(self):
        return self.u @ np.diag(self.s) @ self.v

    def dhat(self):
        # estimate the ranks for shared and specific factors
        d = self.s.shape[0]
        m1hat = self.m1hat()
        m2hat = self.m2hat()
        temp1 = trunsvd(m1hat, d)
        temp2 = trunsvd(m2hat, d)
        d1 = np.sum(temp1[1] >= self.lam * self.gam)
        d2 = np.sum(temp2[1] >= self.lam * self.gam)
        d = self.s.shape[0]
        d = d1 + d2 - d
        d1 = d1 - d
        d2 = d2 - d
        return [d, d1, d2]

    def refresh(self):
        # refresh the estimator u,s,v
        self.u = np.random.randn(2 * self.n1, self.k) / np.power(self.k, 1 / 4)
        self.s = np.ones(self.k)
        self.v = np.random.randn(self.k, self.n2) / np.power(self.k, 1 / 4)

    def factor_loadings(self):
        # return the factor and factor loadings with estimated ranks and matrices
        m1hat = self.m1hat()
        m2hat = self.m2hat()
        k, k1, k2 = self.dhat()
        return get_factor_loadings(m1hat, m2hat, k, k1, k2)


# the estimation of mhat with known ranks, the step 2 estimator
class matcom_oracle:
    def __init__(self, x, w, k, k1, k2, eta=1.0, threshold=10**-5):
        # similarly definition of x, w, eta, threshold, and k, k1, k2 the shared and specific factor's rank
        np.random.seed(2024)
        n1, n2 = x.shape
        self.x = x
        self.w = w
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.eta = eta
        self.threshold = threshold
        self.n1 = n1
        self.n2 = n2
        self.u1 = np.random.randn(n1, (k + k1)) / np.power((k + k1), 1 / 4)
        self.u2 = np.random.randn(n1, (k + k2)) / np.power((k + k2), 1 / 4)
        self.v = np.random.randn((k + k1 + k2), n2) / np.power((k + k1 + k2), 1 / 4)
        # similalry definition for the factor loadings and factors as shown in function get_factor_loadings()

    def loss(self):
        # compute loss
        hat1 = self.u1 @ self.v[: (self.k + self.k1), :]
        hat2 = self.u2 @ self.v[-(self.k + self.k2) :, :]
        loss_temp = loss(self.x, self.w, hat1, hat2, self.eta)
        return loss_temp

    def update_onestep(self, gam=0.1):
        # update factor and factor_loading, gam the learning rate parameter
        hat1 = self.u1 @ self.v[: (self.k + self.k1), :]
        hat2 = self.u2 @ self.v[-(self.k + self.k2) :, :]
        dev, loss = lossd(self.x, self.w, hat1, hat2, self.eta)
        tempu1 = dev[: self.n1, :] @ self.v[: (self.k + self.k1), :].T
        tempu2 = dev[self.n1 :, :] @ self.v[-(self.k + self.k2) :, :].T
        tempv = np.concatenate(
            (self.u1.T @ dev[: self.n1, :], np.zeros((self.k2, self.n2))), axis=0
        ) + np.concatenate(
            (np.zeros((self.k1, self.n2)), self.u2.T @ dev[self.n1 :, :]), axis=0
        )
        self.u1 -= gam * tempu1 / self.n2
        self.u2 -= gam * tempu2 / self.n2
        self.v -= gam * tempv / self.n1
        return loss

    def estimate(self, gam=0.1, max_iter=10000):
        # estimate factor and factor loadings
        loss = np.array([])
        loss = np.append(loss, self.update_onestep(gam))
        loss = np.append(loss, self.update_onestep(gam))
        loss = np.append(loss, self.update_onestep(gam))
        i = 2
        while (np.abs(loss[i] - np.min(loss[max(i - 100, 0) : i])) > self.threshold) & (
            i < max_iter
        ):
            loss = np.append(loss, self.update_onestep(gam))
            i += 1
        self.loss = loss

    def m1hat(self):
        return self.u1 @ self.v[: (self.k + self.k1), :]

    def m2hat(self):
        return self.u2 @ self.v[-(self.k + self.k2) :, :]

    def mhat(self):
        return np.concatenate((self.m1hat(), self.m2hat()), axis=0)

    def etahat(self):
        # give the uncorrected's eta estimation
        sigma2 = np.mean((self.x - self.m2hat()) ** 2 * self.w) / np.mean(self.w)
        self.eta = 1 / sigma2
        return self.eta


# the two step estimator, which combine the two estimator above
class matest:
    def __init__(self, x, w, k, penalty, lam, gam, mu, eta=1.0, threshold=10**-5):
        # similarly definition of parameter in matcom class
        n1, n2 = x.shape
        self.x = x
        self.w = w
        self.k = k
        self.lam = lam
        self.gam = gam
        self.mu = mu
        self.eta = eta
        self.penalty = penalty
        self.threshold = threshold
        self.n1 = n1
        self.n2 = n2

    def first_step(self, iter=10, max_iter=10000):
        first_step = matcom(
            self.x,
            self.w,
            self.k,
            self.penalty,
            self.lam,
            self.gam,
            self.mu,
            self.eta,
            self.threshold,
        )
        first_step.estimate_pre(iter)
        # use 10 step pre training
        first_step.estimate_trunc(max_iter)
        # use rank truncated estimator
        self.kco, self.kt, self.km = first_step.dhat()
        self.m1 = first_step.m1hat()
        self.m2 = first_step.m2hat()
        # get the ranks estimator and matrix estimator from step 1

    def second_step(self, gam=0.1, max_iter=10000):
        if (self.kco + self.kt + self.km) == 0:
            self.kco = 1
            temp = np.random.randn(2 * self.n1, 1) @ np.random.randn(1, self.n2)
            self.m1 = temp[: self.n1, :]
            self.m2 = temp[self.n1 :, :]
        second_step = matcom_oracle(
            self.x, self.w, self.kco, self.kt, self.km, self.eta, gam * self.threshold
        )
        # use step 1's rank estimator as the parameter for step 2
        second_step.u1, second_step.u2, second_step.v = get_factor_loadings(
            self.m1_, self.m2_, self.kco, self.kt, self.km
        )
        # use the step 1's estimator for the initial value of step 2
        second_step.estimate(gam, max_iter)
        self.m1 = second_step.m1hat()
        self.m2 = second_step.m2hat()
        # get the step 2's estimator

    def estimate(self, iter=10, gam=0.1, max_iter=10000):
        self.first_step(iter, max_iter)
        self.m1_ = self.m1
        self.m2_ = self.m2
        self.second_step(gam, max_iter)
        # offer the estimation

    def lk(self):
        m1hat = self.m1
        m2hat = self.m2
        n = self.n1 * self.n2
        n_obs = np.sum(self.w)
        sigmahat = np.sum((m2hat - self.x) ** 2 * self.w) / n_obs
        pihat = 1 / (1 + np.exp(-m1hat))
        pihat = truncate(pihat, 1 - 10 ** (-10), 10 ** (-10))
        l_w = np.sum(self.w * np.log(pihat) + (1 - self.w) * np.log(1 - pihat))
        l_w = -1 * l_w
        l_x = n_obs / 2 * (1 + np.log(2 * np.pi) + np.log(sigmahat))
        meanw = n_obs / n
        free1 = (self.kco + self.kt) * (
            self.n1 + self.n2 - self.kco - self.kt
        ) - self.kco * self.n2 / 2
        free2 = (self.kco + self.km) * (
            self.n1 + self.n2 - self.kco - self.km
        ) - self.kco * self.n2 / 2
        return (l_w, l_x, meanw, free1 + free2)
        # give the likelihood for information criterion, the l_w, l_x as the likelihood for w and x's part, meanw the observation rate, free1 + free2 the freedom of total parameters

    def etahat(self, gam=0.1, max_iter=10000):
        # use eta correction step for the estimator of optimal eta, then do the re-estimator of mhat with the optimal weight
        u1, u2, v = get_factor_loadings(self.m1, self.m2, self.kco, self.kt, self.km)
        v = v.T
        temp = np.zeros_like(u1)
        temp[:, : self.kco] = u1[:, self.kt :]
        temp[:, self.kco :] = u1[:, : self.kt]
        u1 = temp
        temp = np.zeros_like(v)
        temp[:, : self.kco] = v[:, self.kt : (self.kt + self.kco)]
        temp[:, self.kco : (self.kt + self.kco)] = v[:, : self.kt]
        v = temp
        pibar = np.mean(self.w)
        gamthe = np.zeros((self.kco + self.kt + self.km, self.kco + self.kt + self.km))
        if self.kco + self.kt > 0:
            gamthe[: self.kco + self.kt, : self.kco + self.kt] = (
                pibar * (1 - pibar) * u1.T @ u1
            )
        gamm = np.zeros((self.kco + self.kt + self.km, self.kco + self.kt + self.km))
        temp = pibar * self.eta * u2.T @ u2
        if self.kco > 0:
            gamm[: self.kco, : self.kco] = temp[: self.kco, : self.kco]
        if self.km > 0:
            gamm[: self.kco, self.kco + self.kt :] = temp[: self.kco, self.kco :]
        if self.km > 0:
            gamm[self.kco + self.kt :, : self.kco] = temp[self.kco :, : self.kco]
        if self.km > 0:
            gamm[self.kco + self.kt :, self.kco + self.kt :] = temp[
                self.kco :, self.kco :
            ]
        gamma = np.linalg.inv(gamthe + gamm) @ gamm
        tr1 = np.trace(gamma)
        tr2 = np.trace(gamma @ gamma)
        sigma2 = (
            np.sum(self.w * (self.x - self.m2) ** 2) + self.n2 * (tr2 - tr1) / self.eta
        ) / (
            np.sum(self.w) - self.n1 * (self.kco + self.km) - self.n2 * (2 * tr1 - tr2)
        )
        print(sigma2)
        self.eta = 1 / sigma2
        # get the optimal eta
        if (self.kco + self.kt + self.km) == 0:
            self.kco = 1
            temp = np.random.randn(2 * self.n1, 1) @ np.random.randn(1, self.n2)
            self.m1 = temp[: self.n1, :]
            self.m2 = temp[self.n1 :, :]
        second_step = matcom_oracle(
            self.x, self.w, self.kco, self.kt, self.km, self.eta, gam * self.threshold
        )
        second_step.u1, second_step.u2, second_step.v = get_factor_loadings(
            self.m1_, self.m2_, self.kco, self.kt, self.km
        )
        second_step.estimate(gam, max_iter)
        self.m1 = second_step.m1hat()
        self.m2 = second_step.m2hat()
        return self.eta
