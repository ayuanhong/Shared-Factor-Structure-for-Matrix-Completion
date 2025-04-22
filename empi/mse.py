from pathlib import Path

current_file = Path(__file__).resolve()
temp_path = str(current_file.parent) + "/"
print(temp_path)

import numpy as np
import pandas as pd
from numba import vectorize


@vectorize
def truncate(x, a, b):
    if x <= a:
        if x >= b:
            return x
        else:
            return b
    else:
        return a


# Load the data
x1 = pd.read_csv("u1.test", sep="\t", header=None)
x2 = pd.read_csv("u2.test", sep="\t", header=None)
x3 = pd.read_csv("u3.test", sep="\t", header=None)
x4 = pd.read_csv("u4.test", sep="\t", header=None)
x5 = pd.read_csv("u5.test", sep="\t", header=None)
x1, x2, x3, x4, x5 = x1.values, x2.values, x3.values, x4.values, x5.values
test = [x1, x2, x3, x4, x5]
result = np.zeros((5, 5))
rank = np.zeros((5, 5))
path = ["1/", "2/", "3/", "4/", "5/"]
path = [temp_path + i for i in path]


def mse(x, w, m):
    m = truncate(m, 5, 1)
    return np.sum(w * (x - m) ** 2) / np.sum(w)


for i in range(5):
    x_test = np.zeros((943, 1682))
    w_test = np.zeros((943, 1682))
    temp = test[i]
    for j in range(temp.shape[0]):
        x_test[temp[j, 0] - 1, temp[j, 1] - 1] = temp[j, 2]
        w_test[temp[j, 0] - 1, temp[j, 1] - 1] = 1
    sh_m_s = pd.read_csv(path[i] + "sh_m_s.csv", header=None).values
    sh_m_u = pd.read_csv(path[i] + "sh_m_u.csv", header=None).values
    sh_m_v = pd.read_csv(path[i] + "sh_m_v.csv", header=None).values
    sh_op_m_s = pd.read_csv(path[i] + "sh_op_m_s.csv", header=None).values
    sh_op_m_u = pd.read_csv(path[i] + "sh_op_m_u.csv", header=None).values
    sh_op_m_v = pd.read_csv(path[i] + "sh_op_m_v.csv", header=None).values
    mht_m_s = pd.read_csv(path[i] + "mht_m_s.csv", header=None).values
    mht_m_u = pd.read_csv(path[i] + "mht_m_u.csv", header=None).values
    mht_m_v = pd.read_csv(path[i] + "mht_m_v.csv", header=None).values
    nw_m_s = pd.read_csv(path[i] + "nw_m_s.csv", header=None).values
    nw_m_u = pd.read_csv(path[i] + "nw_m_u.csv", header=None).values
    nw_m_v = pd.read_csv(path[i] + "nw_m_v.csv", header=None).values
    mcw_m_s = pd.read_csv(path[i] + "mcw_m_s.csv", header=None).values
    mcw_m_u = pd.read_csv(path[i] + "mcw_m_u.csv", header=None).values
    mcw_m_v = pd.read_csv(path[i] + "mcw_m_v.csv", header=None).values
    mcw_t_s = pd.read_csv(path[i] + "mcw_t_s.csv", header=None).values
    sh_m_s = sh_m_s.flatten()
    sh_op_m_s = sh_op_m_s.flatten()
    mht_m_s = mht_m_s.flatten()
    nw_m_s = nw_m_s.flatten()
    mcw_m_s = mcw_m_s.flatten()
    mcw_t_s = mcw_t_s.flatten()
    sh_m = sh_m_u @ np.diag(sh_m_s) @ sh_m_v
    sh_op_m = sh_op_m_u @ np.diag(sh_op_m_s) @ sh_op_m_v
    mht_m = mht_m_u @ np.diag(mht_m_s) @ mht_m_v
    nw_m = nw_m_u @ np.diag(nw_m_s) @ nw_m_v
    mcw_m = mcw_m_u @ np.diag(mcw_m_s) @ mcw_m_v
    result[i, 0] = mse(x_test, w_test, sh_m)
    result[i, 1] = mse(x_test, w_test, sh_op_m)
    result[i, 2] = mse(x_test, w_test, mht_m)
    result[i, 3] = mse(x_test, w_test, nw_m)
    result[i, 4] = mse(x_test, w_test, mcw_m)
    rank[i, 0] = sh_m_s.shape[0]
    rank[i, 1] = mht_m_s.shape[0]
    rank[i, 2] = nw_m_s.shape[0]
    rank[i, 3] = mcw_m_s.shape[0]
    rank[i, 4] = mcw_t_s.shape[0]

mse = result

mean = np.mean(result, axis=0)
result = result.T
result = np.concatenate((result, mean.reshape(5, 1)), axis=1)
result = np.round(result, 4)
result = result.astype(str)

result_str = result
for i in range(result_str.shape[0]):
    for j in range(1, result_str.shape[1]):
        if len(result_str[i, j].split(".")[1]) != 4:
            temp = 4 - len(result_str[i, j].split(".")[1])
            result_str[i, j] = result_str[i, j] + "0" * temp
result = np.zeros((5, 12))
result = result.astype(str)
for i in range(5):
    result[:, 2 * i + 1] = result_str[:, i]
    for j in range(2, 5):
        result[j, 2 * i + 2] = " " + str(rank[i, j - 1]) + " "
    result[0, 2 * i + 2] = " " + str(rank[i, 0]) + " "
    result[1, 2 * i + 2] = " " + str(rank[i, 0]) + " "

result[:, 0] = ["sh", "osh", "mht", "nw", "mcw"]
result[:, -1] = result_str[:, -1]

pd.DataFrame(result).to_latex(
    temp_path + "result.tex", header=False, index=False, escape=False
)
