from pathlib import Path

current_file = Path(__file__).resolve()
temp_path = str(current_file.parent) + "/"
print(temp_path)

import numpy as np
import pandas as pd

# Load the data
x1 = pd.read_csv("u1.test", sep="\t", header=None)
x2 = pd.read_csv("u2.test", sep="\t", header=None)
x3 = pd.read_csv("u3.test", sep="\t", header=None)
x4 = pd.read_csv("u4.test", sep="\t", header=None)
x5 = pd.read_csv("u5.test", sep="\t", header=None)
x1, x2, x3, x4, x5 = x1.values, x2.values, x3.values, x4.values, x5.values
test = [x1, x2, x3, x4, x5]
path = ["1/", "2/", "3/", "4/", "5/"]
path = [temp_path + i for i in path]
data = []
sh_m = []
sh_m_op = []
mht_m = []
nw_m = []
mcw_m = []
for i in range(5):
    temp = test[i]
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
    sh_m.append([sh_m_u, np.diag(sh_m_s), sh_m_v])
    sh_m_op.append([sh_op_m_u, np.diag(sh_op_m_s), sh_op_m_v])
    mht_m.append([mht_m_u, np.diag(mht_m_s), mht_m_v])
    nw_m.append([nw_m_u, np.diag(nw_m_s), nw_m_v])
    mcw_m.append([mcw_m_u, np.diag(mcw_m_s), mcw_m_v])
    rate_data = {i: dict() for i in range(943)}
    for j in range(temp.shape[0]):
        # 这是第j个用户的东西
        rate_data[temp[j, 0] - 1][temp[j, 1] - 1] = temp[j, 2]
    data.append(rate_data)

m_data = [sh_m, sh_m_op, mht_m, nw_m, mcw_m]

# 把rate_data中没有元素的数据去掉
for i in range(5):
    for j in range(943):
        if len(data[i][j]) == 0:
            data[i].pop(j)

rank = np.zeros((5, 5))
den = np.zeros(5)
for i in range(5):
    for j in data[i].keys():
        real = np.array(list(data[i][j].values()))
        den[i] += np.sum(real)
        for k in range(5):
            tsh = m_data[k][i][0][j] @ m_data[k][i][1] @ m_data[k][i][2]
            tsh = tsh[list(data[i][j].keys())]
            temp = tsh.argsort()[::-1]
            osh = np.zeros(tsh.shape[0])
            osh[temp] = np.arange(tsh.shape[0]) / tsh.shape[0]
            rank[i, k] += np.sum(real * osh)
    for k in range(5):
        rank[i, k] /= den[i]
rank1 = rank

true = np.zeros(5)
den = np.zeros(5)
for i in range(5):
    for j in data[i].keys():
        real = np.array(list(data[i][j].values()))
        temp = real.argsort()[::-1]
        osh = np.zeros(real.shape[0])
        osh[temp] = np.arange(real.shape[0]) / real.shape[0]
        true[i] += np.sum(real * osh)
        den[i] += np.sum(real)
    true[i] /= den[i]

rank = np.concatenate((rank1, true.reshape(5, 1)), axis=1)
rank = rank.T
mean = np.mean(rank, axis=1)
rank = np.concatenate((rank, mean.reshape(6, 1)), axis=1)
rank = np.round(rank, 4)
rank = rank.astype(str)

result_str = rank
for i in range(result_str.shape[0]):
    for j in range(1, result_str.shape[1]):
        if len(result_str[i, j].split(".")[1]) != 4:
            temp = 4 - len(result_str[i, j].split(".")[1])
            result_str[i, j] = result_str[i, j] + "0" * temp
rank = result_str

method = np.array(["sh", "osh", "MHT", "NW", "MCW", "real order"])
rank = np.concatenate((method.reshape(6, 1), rank), axis=1)
pd.DataFrame(rank).to_latex(
    temp_path + "rank.tex", header=False, index=False, escape=False
)
