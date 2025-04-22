from pathlib import Path
import importlib.util

current_file = Path(__file__).resolve()
shfactor_path = current_file.parent.parent / "method" / "shfactor.py"
spec = importlib.util.spec_from_file_location("shfactor", str(shfactor_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
globals().update({k: v for k, v in vars(module).items() if not k.startswith("_")})
temp_path = str(current_file.parent) + "/"

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import math


def one_simu_t5(seed):
    result = np.zeros((lams.shape[0], 19))
    np.random.seed(seed)
    u1 = np.random.randn(n1, k)
    u2 = np.random.randn(n1, k)
    v = np.random.randn(k, n2)
    v_in = np.random.randn(n2 * (k1 + k2)).reshape((k1 + k2, n2))
    v_in = v_in
    m1 = (u1 @ v + np.random.randn(n1, k1) @ v_in[:k1, :]) / np.power((k + k1), 1 / 2)
    m2 = (u2 @ v + np.random.randn(n1, k2) @ v_in[k1:, :]) / np.power((k + k2), 1 / 2)
    m1 = m1 - mp
    m2 = m2 + mc
    pi = 1 / (1 + np.exp(-m1))
    w = np.random.rand(n1, n2) < pi
    w = w + 0
    x = m2 + np.random.standard_t(5, (n1, n2)) * math.sqrt(3 / 5)
    m = np.concatenate((m1, m2), axis=0)
    for i in range(lams.shape[0]):
        hard = matest(x, w, int(np.sqrt(min(n1, n2))), "mcp", lams[i], 1.5, 1.1)
        hard.estimate()
        lk = hard.lk()
        mse1 = np.mean((m1 - hard.m1) ** 2)
        mse2 = np.mean((m2 - hard.m2) ** 2)
        hard.etahat()
        # hard.second_step()
        lk2 = hard.lk()
        mse3 = np.mean((m1 - hard.m1) ** 2)
        mse4 = np.mean((m2 - hard.m2) ** 2)
        result[i, :] = np.array(
            [
                seed,
                lams[i],
                k + 1,
                k1,
                k2,
                hard.kco,
                hard.kt,
                hard.km,
                lk[0],
                lk[1],
                lk[2],
                lk[3],
                mse1,
                mse2,
                lk2[0],
                lk2[1],
                hard.eta,
                mse3,
                mse4,
            ]
        )
    pd.DataFrame(result).to_csv(path, index=False, header=False, mode="a")
    return result


def one_simu_t9(seed):
    result = np.zeros((lams.shape[0], 19))
    np.random.seed(seed)
    u1 = np.random.randn(n1, k)
    u2 = np.random.randn(n1, k)
    v = np.random.randn(k, n2)
    v_in = np.random.randn(n2 * (k1 + k2)).reshape((k1 + k2, n2))
    v_in = v_in
    m1 = (u1 @ v + np.random.randn(n1, k1) @ v_in[:k1, :]) / np.power((k + k1), 1 / 2)
    m2 = (u2 @ v + np.random.randn(n1, k2) @ v_in[k1:, :]) / np.power((k + k2), 1 / 2)
    m1 = m1 - mp
    m2 = m2 + mc
    pi = 1 / (1 + np.exp(-m1))
    w = np.random.rand(n1, n2) < pi
    w = w + 0
    x = m2 + np.random.standard_t(9, (n1, n2)) * math.sqrt(7 / 9)
    m = np.concatenate((m1, m2), axis=0)
    for i in range(lams.shape[0]):
        hard = matest(x, w, int(np.sqrt(min(n1, n2))), "mcp", lams[i], 1.5, 1.1)
        hard.estimate()
        lk = hard.lk()
        mse1 = np.mean((m1 - hard.m1) ** 2)
        mse2 = np.mean((m2 - hard.m2) ** 2)
        hard.etahat()
        # hard.second_step()
        lk2 = hard.lk()
        mse3 = np.mean((m1 - hard.m1) ** 2)
        mse4 = np.mean((m2 - hard.m2) ** 2)
        result[i, :] = np.array(
            [
                seed,
                lams[i],
                k + 1,
                k1,
                k2,
                hard.kco,
                hard.kt,
                hard.km,
                lk[0],
                lk[1],
                lk[2],
                lk[3],
                mse1,
                mse2,
                lk2[0],
                lk2[1],
                hard.eta,
                mse3,
                mse4,
            ]
        )
    pd.DataFrame(result).to_csv(path, index=False, header=False, mode="a")
    return result


n1 = n2 = 500
mp = 2
mc = 1
lams = np.linspace(0.49, 0.58, 21)
sigma = np.sqrt(1)
path = temp_path + "data_500_t5/result_500_t5_low.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )

mp = 1.5
lams = np.linspace(0.45, 0.53, 21)
path = temp_path + "data_500_t5/result_500_t5_mid.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )

mp = 1
lams = np.linspace(0.44, 0.55, 21)
path = temp_path + "data_500_t5/result_500_t5_hig.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t5)(i) for i in range(50)
    )


n1 = n2 = 500
mp = 2
mc = 1
lams = np.linspace(0.49, 0.58, 21)
sigma = np.sqrt(1)
path = temp_path + "data_500_t9/result_500_t9_low.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )

mp = 1.5
lams = np.linspace(0.45, 0.53, 21)
path = temp_path + "data_500_t9/result_500_t9_mid.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )

mp = 1
lams = np.linspace(0.44, 0.55, 21)
path = temp_path + "data_500_t9/result_500_t9_hig.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_t9)(i) for i in range(50)
    )
