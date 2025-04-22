from pathlib import Path
import importlib.util

current_file = Path(__file__).resolve()
compare_path = current_file.parent.parent / "method" / "compare_empi.py"
spec = importlib.util.spec_from_file_location("compare_empi", str(compare_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
globals().update({k: v for k, v in vars(module).items() if not k.startswith("_")})
compare_path = current_file.parent.parent / "method" / "shfactor.py"
spec = importlib.util.spec_from_file_location("shfactor", str(compare_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
globals().update({k: v for k, v in vars(module).items() if not k.startswith("_")})

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

temp_path = str(current_file.parent.parent)
print(temp_path)


def get_sh(lam):
    hard = matest(x_train, w_train, int(np.sqrt(min(n1, n2))), "mcp", lam, 1.5, 1.1)
    hard.estimate()
    lk = hard.lk()
    mest = truncate(hard.m2, 5, 1)
    pest = 1 / (1 + np.exp(-truncate(hard.m1, 10, -10)))
    mse1 = np.mean(w_test * (mest - x_test) ** 2) / np.mean(w_test)
    loss2 = -np.mean(w_test * np.log(pest) + (1 - w_test) * np.log(1 - pest))
    result = np.array(lam)
    result = np.append(result, np.array(lk))
    result = np.append(result, np.array([hard.kco, hard.kt, hard.km, mse1, loss2]))
    result = result.reshape(1, -1)
    pd.DataFrame(result).to_csv(path, sep=",", index=False, mode="a", header=False)
    return result


def get_result_mao_pro(lam):
    n1, n2 = w_train.shape
    hard = mao(w_train, min(n1, n2), lam)
    hard.estimate_pre(10)
    hard.estimate_truncate()
    result = np.array(lam)
    result = np.append(result, np.array(hard.aic()))
    result = result.reshape(1, -1)
    pd.DataFrame(result).to_csv(path, sep=",", index=False, mode="a", header=False)
    return result


def get_result_mao_m(lam):
    n1, n2 = w_train.shape
    hard = mat_wei(x_train, w_train * wei_ml, min(n1, n2), lam, max_wei)
    result = np.array(lam)
    result = np.append(result, np.array(hard.cv()))
    result = result.reshape(1, -1)
    pd.DataFrame(result).to_csv(path, sep=",", index=False, mode="a", header=False)
    return result


now_set = range(1, 6, 1)
now_set = [str(x) for x in now_set]
now_set = ["1"]
now = "1"

for now in now_set:
    train = pd.read_csv(temp_path + "/empi/data/u" + now + ".base", sep="\t")
    test = pd.read_csv(temp_path + "/empi/data/u" + now + ".test", sep="\t")
    train = np.array(train)
    test = np.array(test)
    x_train = np.zeros((943, 1682))
    w_train = np.zeros((943, 1682))
    for i in range(train.shape[0]):
        x_train[train[i, 0] - 1, train[i, 1] - 1] = train[i, 2]
        w_train[train[i, 0] - 1, train[i, 1] - 1] = 1

    x_test = np.zeros((943, 1682))
    w_test = np.zeros((943, 1682))
    for i in range(test.shape[0]):
        x_test[test[i, 0] - 1, test[i, 1] - 1] = test[i, 2]
        w_test[test[i, 0] - 1, test[i, 1] - 1] = 1

    n1, n2 = x_train.shape

    # use Shared method
    print(temp_path)
    path = temp_path + "/empi/" + now + "/sh_lam.csv"
    print(path)
    np.random.seed(2024)
    lams = np.linspace(0.6, 0.75, 16)
    result = Parallel(n_jobs=-1, batch_size=1)(delayed(get_sh)(num) for num in lams)

    x = pd.read_csv(path, header=None)
    x = np.array(x)
    ic = x[:, 1] + x[:, 2] + 0.1 * np.log(943 * 1682) * x[:, 4]
    loc = np.argmin(ic)
    print(x[loc, :])

    lam = x[loc, 0]
    hard = matest(x_train, w_train, int(np.sqrt(min(n1, n2))), "mcp", lam, 1.5, 1.1)
    hard.estimate()
    km = hard.kco + hard.km
    kt = hard.kco + hard.kt
    temp = trunsvd(hard.m2, km)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/sh_m_u.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/sh_m_s.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/sh_m_v.csv", sep=",", index=False, header=False
    )
    temp = trunsvd(hard.m1, kt)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/sh_t_u.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/sh_t_s.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/sh_t_v.csv", sep=",", index=False, header=False
    )
    hard.etahat()
    km = hard.kco + hard.km
    kt = hard.kco + hard.kt
    temp = trunsvd(hard.m2, km)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/sh_op_m_u.csv",
        sep=",",
        index=False,
        header=False,
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/sh_op_m_s.csv",
        sep=",",
        index=False,
        header=False,
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/sh_op_m_v.csv",
        sep=",",
        index=False,
        header=False,
    )
    temp = trunsvd(hard.m1, kt)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/sh_op_t_u.csv",
        sep=",",
        index=False,
        header=False,
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/sh_op_t_s.csv",
        sep=",",
        index=False,
        header=False,
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/sh_op_t_v.csv",
        sep=",",
        index=False,
        header=False,
    )

    # use MHT method
    path = temp_path + "/empi/" + now + "/mht_lam.csv"
    np.random.seed(2024)

    wei_ml = np.ones((943, 1682)) * w_train
    max_wei = 1

    lams = np.linspace(0.01, 0.1, 16)
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(get_result_mao_m)(num) for num in lams
    )

    x = pd.read_csv(path, header=None)
    x = np.array(x)
    loc = np.argmin(x[:, 1])
    print(x[loc, :])

    lam = x[loc, 0]
    hard = mat_wei(x_train, w_train * wei_ml, int(np.sqrt(min(n1, n2))), lam, max_wei)
    hard.estimate(10)
    hard.estimate_truncate()
    r = hard.rank()
    temp = trunsvd(hard.m, r)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/mht_m_u.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/mht_m_s.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/mht_m_v.csv", sep=",", index=False, header=False
    )

    # use NW method
    path = temp_path + "/empi/" + now + "/nw_lam.csv"
    np.random.seed(2024)

    row_mean = np.mean(w_train, axis=1)
    col_mean = np.mean(w_train, axis=0)
    pi = row_mean.reshape(943, 1) @ col_mean.reshape(1, 1682)
    pi = truncate(pi, 1, 10**-5)
    wei_ml = 1 / pi
    wei_ml = w_train * wei_ml
    wei_ml = wei_ml / np.mean(wei_ml) * np.mean(w_train)
    max_wei = np.quantile(wei_ml, 0.999)
    wei_ml = truncate(wei_ml, max_wei, 0)
    print(max_wei)

    lams = np.linspace(0.01, 0.1, 16)
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(get_result_mao_m)(num) for num in lams
    )

    x = pd.read_csv(path, header=None)
    x = np.array(x)
    loc = np.argmin(x[:, 1])
    print(x[loc, :])

    lam = x[loc, 0]
    hard = mat_wei(x_train, w_train * wei_ml, int(np.sqrt(min(n1, n2))), lam, max_wei)
    hard.estimate(10)
    hard.estimate_truncate()
    r = hard.rank()
    temp = trunsvd(hard.m, r)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/nw_m_u.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/nw_m_s.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/nw_m_v.csv", sep=",", index=False, header=False
    )

    # use  MCW method
    path = temp_path + "/empi/" + now + "/mcw_t_lam.csv"
    np.random.seed(2024)
    lams = np.linspace(0.02, 0.17, 16)
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(get_result_mao_pro)(num) for num in lams
    )

    x = pd.read_csv(path, header=None)
    x = np.array(x)
    loc = np.argmin(x[:, 1])
    print(x[loc, :])

    lam = x[loc, 0]
    hard = mao(w_train, int(np.sqrt(min(n1, n2))), lam)
    hard.estimate_pre(10)
    hard.estimate_truncate()
    r = hard.rank()
    temp = trunsvd(hard.hat(), r)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/mcw_t_u.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/mcw_t_s.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/mcw_t_v.csv", sep=",", index=False, header=False
    )

    wei_ml = 1 + np.exp(-hard.hat())
    max_wei = np.quantile(wei_ml, 0.999)
    min_wei = np.quantile(wei_ml, 0.001)
    wei_ml = truncate(wei_ml, max_wei, min_wei)
    wei_ml = w_train * wei_ml
    wei_ml = wei_ml / np.mean(wei_ml) * np.mean(w_train)
    max_wei = np.quantile(wei_ml, 0.999)
    wei_ml = truncate(wei_ml, max_wei, 0)
    print(max_wei)

    path = temp_path + "/empi/" + now + "/mcw_m_lam.csv"
    np.random.seed(2024)
    lams = np.linspace(0.01, 0.1, 16)
    result = Parallel(n_jobs=-1, batch_size=1)(
        delayed(get_result_mao_m)(num) for num in lams
    )

    x = pd.read_csv(path, header=None)
    x = np.array(x)
    loc = np.argmin(x[:, 1])
    print(x[loc, :])

    lam = x[loc, 0]
    hard = mat_wei(x_train, w_train * wei_ml, int(np.sqrt(min(n1, n2))), lam, max_wei)
    hard.estimate(10)
    hard.estimate_truncate()
    r = hard.rank()
    temp = trunsvd(hard.m, r)
    pd.DataFrame(temp[0]).to_csv(
        temp_path + "/empi/" + now + "/mcw_m_u.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[1]).to_csv(
        temp_path + "/empi/" + now + "/mcw_m_s.csv", sep=",", index=False, header=False
    )
    pd.DataFrame(temp[2]).to_csv(
        temp_path + "/empi/" + now + "/mcw_m_v.csv", sep=",", index=False, header=False
    )
