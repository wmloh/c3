import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_mind_cumregret(base_dir, alg_list, label_list, figsize=(9, 5),
                        titles=None, conf=1):
    fig, ax = plt.subplots(figsize=figsize)

    all_mean_regrets = np.zeros((len(alg_list), 500)) + np.inf
    all_std_regrets = np.zeros((len(alg_list), 500)) + np.inf

    for i, (alg, label) in enumerate(zip(alg_list, label_list)):
        path = os.path.join(base_dir, f"mind_{alg}")

        all_regret = list()
        for seed in sorted(os.listdir(path)):
            pkl_path = os.path.join(path, seed, "results_dict.pkl")

            if not os.path.exists(path):  # results_dict cannot be found
                print(f"{path} not found")
                continue

            with open(pkl_path, "rb") as f:
                results_dict = pickle.load(f)
            all_regret.append(results_dict["cum_regret"][1:])

        all_regret = np.array(all_regret)

        mean_reg = all_regret.mean(axis=0)
        std_reg = all_regret.std(axis=0) / np.sqrt(len(all_regret))

        all_mean_regrets[i] = mean_reg
        all_std_regrets[i] = std_reg

        print(path, (len(mean_reg) - mean_reg[-1]) * 100 / len(mean_reg))

    diff_reg = np.concatenate((np.zeros((len(alg_list), 1)), np.diff(all_mean_regrets)), axis=-1)
    diff_reg -= diff_reg.min(axis=0).reshape(1, -1)

    mean_filter = np.cumsum(diff_reg, axis=-1)
    std_filter = all_std_regrets

    print(mean_filter[:, -1])

    for i, (alg, label) in enumerate(zip(alg_list, label_list)):
        mean_reg = mean_filter[i]
        std_reg = std_filter[i]

        ax.plot(mean_reg, label=label, alpha=0.8)
        ax.fill_between(np.arange(len(mean_reg)), mean_reg - conf * std_reg, mean_reg + conf * std_reg, alpha=0.2)
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Cumulative regret")

    if titles is None:
        ax.set_title("MIND")
    else:
        ax.set_title(titles)
    ax.legend()

    plt.tight_layout()


def plot_mind_time(base_dir, alg_list, label_list, figsize=(9, 5),
                   titles=None, conf=1):
    fig, ax = plt.subplots(figsize=figsize)

    all_mean_regrets = np.zeros((len(alg_list), 500)) + np.inf
    all_std_regrets = np.zeros((len(alg_list), 500)) + np.inf

    for i, (alg, label) in enumerate(zip(alg_list, label_list)):
        path = os.path.join(base_dir, f"mind_{alg}")

        all_regret = list()
        for seed in sorted(os.listdir(path)):
            pkl_path = os.path.join(path, seed, "results_dict.pkl")

            if not os.path.exists(path):  # results_dict cannot be found
                print(f"{path} not found")
                continue

            with open(pkl_path, "rb") as f:
                results_dict = pickle.load(f)
            all_regret.append(results_dict["time_steps"])

        all_regret = np.array(all_regret)

        mean_reg = all_regret.mean(axis=0)
        std_reg = all_regret.std(axis=0) / np.sqrt(len(all_regret))

        all_mean_regrets[i] = mean_reg
        all_std_regrets[i] = std_reg

    for i, (alg, label) in enumerate(zip(alg_list, label_list)):
        mean_reg = all_mean_regrets[i]
        std_reg = all_std_regrets[i]

        ax.plot(mean_reg, label=label, alpha=0.8)
        ax.fill_between(np.arange(len(mean_reg)), mean_reg - conf * std_reg, mean_reg + conf * std_reg, alpha=0.2)
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Time elapsed")

    if titles is None:
        ax.set_title("MIND time analysis")
    else:
        ax.set_title(titles)
    ax.legend()

    plt.tight_layout()


if __name__ == "__main__":
    base_dir = "results/"
    plot_mind_cumregret(base_dir, ["C3", "bayeslr", "tt_small", "tt_med", "swlinucb", "dlinucb", "gpforget", "onlinenucb", "onlinents", "crb"],
                        ["$C_3$", "BayesLR", "Two Tower (small)", "Two Tower (large)", "SW-LinUCB", "D-LinUCB",
                         "Gaussian Process", "Online-NUCB", "Online-NTS", "CRB"],
                        titles="MIND", figsize=(6, 4))
    plt.savefig(os.path.join("figures", "MIND_regret_tmlr.png"), dpi=300, pad_inches=0, bbox_inches='tight')

    # plot_mind_time(base_dir, ["C3", "bayeslr", "tt_small", "tt_med", "swlinucb", "dlinucb", "gpforget", "onlinenucb", "onlinents", "crb"],
    #                ["$C_3$", "BayesLR", "Two Tower (small)", "Two Tower (large)", "SW-LinUCB", "D-LinUCB",
    #                 "Gaussian Process", "Online-NUCB", "Online-NTS", "CRB"],
    #                titles="MIND time analysis", figsize=(6, 4))
    # plt.savefig(os.path.join("figures", "MIND_time_tmlr.png"), dpi=300, pad_inches=0, bbox_inches='tight')
