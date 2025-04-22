import numpy as np
import pandas as pd


def write_txt(path, pro):
    x = pd.read_csv(
        "data_500_" + path + "_com/500_result_" + path + "_com_" + pro + ".csv",
        header=None,
    )
    # read the data
    x = np.array(x)
    x = x.reshape(-1, 8)
    # grouped by the model
    grouped = np.array(
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

    y = pd.read_csv(
        "data_500_" + path + "/result_500_" + path + "_" + pro + ".csv", header=None
    )
    y = np.array(y)
    groupedy = y[:, [0, 2, 3, 4]]
    groupedy = np.unique(groupedy, axis=0)
    groupedy = groupedy[np.lexsort(groupedy.T)]
    z = np.zeros((groupedy.shape[0], 10))
    # choose the minimum information criterion lam
    for i in range(len(groupedy)):
        index = np.where(
            (y[:, 0] == groupedy[i, 0])
            & (y[:, 2] == groupedy[i, 1])
            & (y[:, 3] == groupedy[i, 2])
            & (y[:, 4] == groupedy[i, 3])
        )
        index = index[0]
        z[i, 0:3] = groupedy[i, 1:4]
        temp = y[index, :]
        temp = temp[:, [8, 9, 11]]
        ic = temp[:, 0] + temp[:, 1] + 0.25 * np.log(500) * temp[:, 2]
        min_index = np.argmin(ic)
        z[i, 3:10] = y[index[min_index], [18, 17, 13, 12, 5, 6, 7]]

    result = np.zeros((2 * grouped.shape[0], 9))
    for i in range(grouped.shape[0]):
        index = np.where(
            (x[:, 1] == grouped[i, 0])
            & (x[:, 2] == grouped[i, 1])
            & (x[:, 3] == grouped[i, 2])
        )
        index = index[0]
        temp = x[index, :]
        temp = temp[:, [4, 5, 6, 7]]
        mean = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)
        result[2 * i, [3, 4, 5]] = mean[0:3]
        result[2 * i, 8] = mean[3]
        result[2 * i + 1, [3, 4, 5]] = std[0:3]
        result[2 * i + 1, 8] = std[3]
        index = np.where(
            (z[:, 0] == grouped[i, 0])
            & (z[:, 1] == grouped[i, 1])
            & (z[:, 2] == grouped[i, 2])
        )
        index = index[0]
        temp = z[index, :]
        temp = temp[:, 3:7]
        mean = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)
        result[2 * i, [1, 2]] = mean[[0, 2]]
        result[2 * i + 1, [1, 2]] = std[[0, 2]]
        result[2 * i, [6, 7]] = mean[[1, 3]]
        result[2 * i + 1, [6, 7]] = std[[1, 3]]

    min_index = np.zeros(result.shape[0] // 2)
    min_index_1 = np.zeros(result.shape[0] // 2)
    for i in range(result.shape[0] // 2):
        min_index[i] = np.argmin(result[2 * i, 1:6])
        min_index_1[i] = np.argmin(result[2 * i, 6:])
    min_index = min_index.astype(int)
    min_index_1 = min_index_1.astype(int)
    result = np.round(result, 4)

    # transport result to string
    result = result.astype(str)

    grouped = grouped.astype(str)
    for i in range(grouped.shape[0]):
        for j in range(grouped.shape[1]):
            grouped[i, j] = grouped[i, j].split(".")[0]

    # add 0 after result's value
    for i in range(result.shape[0]):
        for j in range(1, result.shape[1]):
            if len(result[i, j].split(".")[1]) != 4:
                temp = 4 - len(result[i, j].split(".")[1])
                result[i, j] = result[i, j] + "0" * temp

    for i in range(result.shape[0] // 2):
        result[2 * i, min_index[i] + 1] = (
            "\\textbf{" + result[2 * i, min_index[i] + 1] + "}"
        )
        result[2 * i, min_index_1[i] + 6] = (
            "\\textbf{" + result[2 * i, min_index_1[i] + 6] + "}"
        )

    # add () for even line
    for i in range(result.shape[0]):
        if i % 2 != 0:
            for j in range(1, result.shape[1]):
                result[i, j] = "(" + result[i, j] + ")"

    for i in range(result.shape[0] // 2):
        result[2 * i, 0] = (
            "\\multirow{2}{*}{"
            + grouped[i, 0]
            + ","
            + grouped[i, 1]
            + ","
            + grouped[i, 2]
            + "}"
        )
        result[2 * i + 1, 0] = ""

    # write result to tex file
    result = pd.DataFrame(result)
    result.to_latex(
        "temp/" + path + "_" + pro + ".tex", index=False, header=False, escape=False
    )
    result.to_csv("temp/" + path + "_" + pro + ".csv", index=False, header=False)


path = ["1", "0.5", "1.5"]
pro = ["hig", "mid", "low"]
for i in range(len(path)):
    for j in range(len(pro)):
        write_txt(path[i], pro[j])
