import numpy as np
import pandas as pd

# Load data
x_1_low = pd.read_csv("data_500_1/result_500_1_low.csv", header=None)
x_1_mid = pd.read_csv("data_500_1/result_500_1_mid.csv", header=None)
x_1_hig = pd.read_csv("data_500_1/result_500_1_hig.csv", header=None)
x_t5_low = pd.read_csv("data_500_t5/result_500_t5_low.csv", header=None)
x_t5_mid = pd.read_csv("data_500_t5/result_500_t5_mid.csv", header=None)
x_t5_hig = pd.read_csv("data_500_t5/result_500_t5_hig.csv", header=None)
x_t9_low = pd.read_csv("data_500_t9/result_500_t9_low.csv", header=None)
x_t9_mid = pd.read_csv("data_500_t9/result_500_t9_mid.csv", header=None)
x_t9_hig = pd.read_csv("data_500_t9/result_500_t9_hig.csv", header=None)

# take a list to load all data
x = [
    x_1_hig,
    x_t5_hig,
    x_t9_hig,
    x_1_mid,
    x_t5_mid,
    x_t9_mid,
    x_1_low,
    x_t5_low,
    x_t9_low,
]

mean1 = np.zeros((9, 9))
sd1 = np.zeros((9, 9))
mean2 = np.zeros((9, 9))
sd2 = np.zeros((9, 9))
mean3 = np.zeros((9, 9))
sd3 = np.zeros((9, 9))
group = np.array(
    [
        [9, 0, 0],
        [5, 2, 2],
        [1, 4, 4],
        [7, 0, 2],
        [5, 0, 4],
        [3, 0, 6],
        [7, 2, 0],
        [5, 4, 0],
        [3, 6, 0],
    ]
)

rate = np.zeros((9, 9))

for i in range(9):
    y = np.array(x[i])
    groupedy = y[:, [0, 2, 3, 4]]
    groupedy = np.unique(groupedy, axis=0)
    groupedy = groupedy[np.lexsort(groupedy.T)]
    z = np.zeros((groupedy.shape[0], 10))
    # choose the lam make IC minimum
    for j in range(len(groupedy)):
        index = np.where(
            (y[:, 0] == groupedy[j, 0])
            & (y[:, 2] == groupedy[j, 1])
            & (y[:, 3] == groupedy[j, 2])
            & (y[:, 4] == groupedy[j, 3])
        )
        index = index[0]
        z[j, 0:3] = groupedy[j, 1:4]
        temp = y[index, :]
        temp = temp[:, [8, 9, 11]]
        ic = temp[:, 0] + temp[:, 1] + 0.25 * np.log(500) * temp[:, 2]
        # compute ic
        min_index = np.argmin(ic)
        z[j, 3:9] = y[index[min_index], [13, 12, 16, 5, 6, 7]]
        z[j, 9] = np.sum(np.abs(z[j, 0:3] - z[j, 6:9]))
        z[j, 9] = z[j, 9] < 0.1
    for j in range(9):
        index = np.where(
            (z[:, 0] == group[j, 0])
            & (z[:, 1] == group[j, 1])
            & (z[:, 2] == group[j, 2])
        )
        mean1[j, i] = np.mean(z[index, 3] * z[index, 9]) / np.mean(z[index, 9])
        sd1[j, i] = np.sqrt(
            np.mean(z[index, 3] ** 2 * z[index, 9]) / np.mean(z[index, 9])
            - (np.mean(z[index, 3] * z[index, 9]) / np.mean(z[index, 9])) ** 2
        )
        mean2[j, i] = np.mean(z[index, 4] * z[index, 9]) / np.mean(z[index, 9])
        sd2[j, i] = np.sqrt(
            np.mean(z[index, 4] ** 2 * z[index, 9]) / np.mean(z[index, 9])
            - (np.mean(z[index, 4] * z[index, 9]) / np.mean(z[index, 9])) ** 2
        )
        mean3[j, i] = np.mean(z[index, 5] * z[index, 9]) / np.mean(z[index, 9])
        sd3[j, i] = np.sqrt(
            np.mean(z[index, 5] ** 2 * z[index, 9]) / np.mean(z[index, 9])
            - (np.mean(z[index, 5] * z[index, 9]) / np.mean(z[index, 9])) ** 2
        )
        rate[j, i] = np.mean(z[index, 9])


mean1 = np.round(mean1, 4)
sd1 = np.round(sd1, 4)
mean2 = np.round(mean2, 4)
sd2 = np.round(sd2, 4)
mean1 = mean1.astype(str)
sd1 = sd1.astype(str)
mean2 = mean2.astype(str)
sd2 = sd2.astype(str)
group = np.array(
    ["9,0,0", "5,2,2", "1,4,4", "7,0,2", "5,0,4", "3,0,6", "7,2,0", "5,4,0", "3,6,0"]
)
result1 = np.zeros((18, 7))
result1 = result1.astype(str)
result2 = np.zeros((18, 7))
result2 = result2.astype(str)
result3 = np.zeros((18, 7))
result3 = result3.astype(str)

for i in range(9):
    result1[2 * i, 0] = "\multirow{2}{*}{" + group[i] + "}"
    result1[2 * i + 1, 0] = " "
    result2[2 * i, 0] = "\multirow{2}{*}{" + group[i] + "}"
    result2[2 * i + 1, 0] = " "
    result3[2 * i, 0] = "\multirow{2}{*}{ " + group[i] + "}"
    result3[2 * i + 1, 0] = " "
    result1[2 * i, 1:4] = mean1[i, 0:3]
    result1[2 * i, 4:7] = mean2[i, 0:3]
    result1[2 * i + 1, 1:4] = sd1[i, 0:3]
    result1[2 * i + 1, 4:7] = sd2[i, 0:3]
    result2[2 * i, 1:4] = mean1[i, 3:6]
    result2[2 * i, 4:7] = mean2[i, 3:6]
    result2[2 * i + 1, 1:4] = sd1[i, 3:6]
    result2[2 * i + 1, 4:7] = sd2[i, 3:6]
    result3[2 * i, 1:4] = mean1[i, 6:9]
    result3[2 * i, 4:7] = mean2[i, 6:9]
    result3[2 * i + 1, 1:4] = sd1[i, 6:9]
    result3[2 * i + 1, 4:7] = sd2[i, 6:9]

result = np.concatenate((result1[:, 0:4], result2[:, 1:4], result3[:, 1:4]), axis=1)

result_str = result
for i in range(result_str.shape[0]):
    for j in range(1, result_str.shape[1]):
        if len(result_str[i, j].split(".")[1]) != 4:
            temp = 4 - len(result_str[i, j].split(".")[1])
            result_str[i, j] = result_str[i, j] + "0" * temp
result = result_str

for i in range(result.shape[0]):
    if i % 2 != 0:
        for j in range(1, result.shape[1]):
            result[i, j] = "(" + result[i, j] + ")"

result = pd.DataFrame(result)
result.to_latex("temp/distribution_mse.tex", index=False, header=False, escape=False)

