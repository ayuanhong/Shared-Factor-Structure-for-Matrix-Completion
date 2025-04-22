from pathlib import Path
import importlib.util

current_file = Path(__file__).resolve()
compare_path = current_file.parent.parent / "method" / "compare_simu.py"
spec = importlib.util.spec_from_file_location("compare_simu", str(compare_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
globals().update({k: v for k, v in vars(module).items() if not k.startswith("_")})
temp_path = str(current_file.parent) + "/"

from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def one_simu_compete(seed):
    np.random.seed(seed)
    u1 = np.random.randn(n1, k)
    u2 = np.random.randn(n1, k)
    v = np.random.randn(k, n2)
    v_in = np.random.randn(n2 * (k1 + k2)).reshape((k1 + k2, n2))
    m1 = (u1 @ v + np.random.randn(n1, k1) @ v_in[:k1, :]) / np.power((k + k1), 1 / 2)
    m2 = (u2 @ v + np.random.randn(n1, k2) @ v_in[k1:, :]) / np.power((k + k2), 1 / 2)
    m1 = m1 - mp
    m2 = m2 + mc
    pi = 1 / (1 + np.exp(-m1))
    w = np.random.rand(n1, n2) < pi
    w = w + 0
    x = m2 + np.random.randn(n1, n2) * sigma
    m = np.concatenate((m1, m2), axis=0)
    result = np.zeros(4)
    hard = matcom_compete(x, w, k + k1 + 1, k + k2 + 1)
    hard.estimate()
    mht = np.mean((m2 - hard.m2hat()) ** 2)
    hard.estimate_pro()
    hard.estimate()
    mwc = np.mean((m2 - hard.m2hat()) ** 2)
    mwcp = np.mean((m1 - hard.m1hat()) ** 2)
    hard.estimate_pro_NW()
    hard.estimate()
    nw = np.mean((m2 - hard.m2hat()) ** 2)
    result = np.array([seed, k + 1, k1, k2, mht, mwc, nw, mwcp])
    pd.DataFrame(result).to_csv(path, index=False, header=False, mode="a")
    return result


n1 = n2 = 500
mp = 2
mc = 1
sigma = np.sqrt(0.5)
path = temp_path + "data_500_0.5_com/result_500_0.5_com_low.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )

mp = 1.5
path = temp_path + "data_500_0.5_com/result_500_0.5_com_mid.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )


mp = 1
path = temp_path + "data_500_0.5_com/result_500_0.5_com_hig.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )


n1 = n2 = 500
mp = 2
mc = 1
sigma = np.sqrt(1)
path = temp_path + "data_500_1_com/result_500_1_com_low.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )

mp = 1.5
path = temp_path + "data_500_1_com/result_500_1_com_mid.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )


mp = 1
path = temp_path + "data_500_1_com/result_500_1_com_hig.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )


n1 = n2 = 500
mp = 2
mc = 1
sigma = np.sqrt(1.5)
path = temp_path + "data_500_1.5_com/result_500_1.5_com_low.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )

mp = 1.5
path = temp_path + "data_500_1.5_com/result_500_1.5_com_mid.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )


mp = 1
path = temp_path + "data_500_1.5_com/result_500_1.5_com_hig.csv"
for id in range(3):
    k = 8 - 4 * id
    k1 = 2 * id
    k2 = 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 2 + 2 * id
    k2 = 0
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
for id in range(3):
    k = 6 - 2 * id
    k1 = 0
    k2 = 2 + 2 * id
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_simu_compete)(i) for i in range(50)
    )
