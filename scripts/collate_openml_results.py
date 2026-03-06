import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from collections import defaultdict


def plot_cumregret(base_dir, label, ax):
    all_regret = list()
    for seed in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, seed, "results_dict.pkl")

        if not os.path.exists(path):  # results_dict cannot be found
            print(f"{path} not found")
            continue

        with open(path, "rb") as f:
            results_dict = pickle.load(f)
        all_regret.append([0] + results_dict["cum_regret"][1:])

    if len(all_regret) == 0:
        ax.plot([], [])
        ax.fill_between([], [], [])
        return

    all_regret = np.array(all_regret)

    mean_reg = all_regret.mean(axis=0)
    std_reg = all_regret.std(axis=0) / np.sqrt(len(all_regret))

    if VERBOSE:
        print(base_dir, (len(mean_reg) - mean_reg[-1]) * 100 / len(mean_reg))

    ax.plot(mean_reg, label=label, alpha=0.8)
    ax.fill_between(np.arange(len(mean_reg)), mean_reg - 1.96 * std_reg, mean_reg + 1.96 * std_reg, alpha=0.2)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Cumulative regret")


def plot_regret(base_dir, label, ax, static=False, w=50):
    all_regret = list()
    for seed in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, seed, "results_dict.pkl")

        if not os.path.exists(path):  # results_dict cannot be found
            print(f"{path} not found")
            continue

        with open(path, "rb") as f:
            results_dict = pickle.load(f)
        all_regret.append([0] + results_dict["cum_regret"][1:])

    if len(all_regret) == 0:
        ax.plot([], [])
        ax.fill_between([], [], [])
        return

    all_regret = np.array(all_regret)
    all_regret = np.diff(all_regret)

    all_regret = np.stack([np.convolve(q, np.ones(w), 'valid') / w for q in all_regret])

    mean_reg = all_regret.mean(axis=0)
    std_reg = all_regret.std(axis=0) / np.sqrt(len(all_regret))

    if VERBOSE:
        print(base_dir, (1 - mean_reg[-1]) * 100)

    if not static:
        ax.plot(mean_reg, label=label, alpha=0.8)
        ax.fill_between(np.arange(len(mean_reg)), mean_reg - 1.96 * std_reg, mean_reg + 1.96 * std_reg, alpha=0.15)
    else:
        ax.plot(mean_reg, label=label, alpha=0.8, linestyle="--", c="tab:gray")
        ax.fill_between(np.arange(len(mean_reg)), mean_reg - 1.96 * std_reg, mean_reg + 1.96 * std_reg, alpha=0.15, color="tab:gray")

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Average regret")


def plot_time(base_dir, label, ax):
    all_times = list()
    for seed in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, seed, "results_dict.pkl")

        if not os.path.exists(path):  # results_dict cannot be found
            print(f"{path} not found")
            continue

        with open(path, "rb") as f:
            results_dict = pickle.load(f)
        all_times.append([0] + results_dict["time_steps"])

    if len(all_times) == 0:
        ax.plot([], [])
        ax.fill_between([], [], [])
        return

    all_times = np.array(all_times)

    mean_times = all_times.mean(axis=0)
    std_times = all_times.std(axis=0) / np.sqrt(len(all_times))

    progress = np.linspace(0, 1, num=len(mean_times)) * 100

    ax.plot(mean_times, progress, label=label, alpha=0.8)
    ax.fill_betweenx(progress, mean_times - 1.96 * std_times, mean_times + 1.96 * std_times, alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Test progress (%)")


def separate_results(base_dir, datasets, alg_list, label_list, kind, subplot_format, figsize=(9, 5),
                     static_alg_list=[], static_label_list=[], titles=None,
                     bbox_to_anchor=(1, 0.5), rect=(0, 0, 0.83, 1),
                     **kwargs):
    fig, axes = plt.subplots(*subplot_format, figsize=figsize)

    for i, dataset in enumerate(datasets):
        ax = axes[i // 2, i % 2] if subplot_format != (1, 1) else axes
        for alg, label in zip(alg_list, label_list):
            path = os.path.join(base_dir, f"{dataset}/{alg}")

            if kind == "cumregret":
                plot_cumregret(path, label, ax, **kwargs)
            elif kind == "regret":
                plot_regret(path, label, ax, **kwargs)
            elif kind == "time":
                plot_time(path, label, ax, **kwargs)
            else:
                raise ValueError(f"{kind} is not a valid kind")

        for alg, label in zip(static_alg_list, static_label_list):
            path = os.path.join(base_dir, f"{dataset}/{alg}")

            if kind == "cumregret":
                plot_cumregret(path, label, ax, **kwargs)
            elif kind == "regret":
                plot_regret(path, label, ax, static=True, **kwargs)
            elif kind == "time":
                plot_time(path, label, ax, **kwargs)
            else:
                raise ValueError(f"{kind} is not a valid kind")

        if titles is None:
            ax.set_title(dataset)
        else:
            ax.set_title(titles[i])
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=bbox_to_anchor, frameon=False)
    plt.tight_layout(rect=rect)


def get_improvement(base_dir, datasets, alg_list, c3_index=0):

    result = {alg: 0 for alg in alg_list}
    dataset_acc = {d: dict() for d in datasets}

    for alg in alg_list:
        overall_acc = list()
        for i, dataset in enumerate(datasets):

            dir_path = os.path.join(base_dir, f"{dataset}_{alg}")
            all_regret = list()

            for seed in sorted(os.listdir(dir_path)):
                path = os.path.join(dir_path, seed, "results_dict.pkl")

                if not os.path.exists(path):  # results_dict cannot be found
                    print(f"{path} not found")
                    continue

                with open(path, "rb") as f:
                    results_dict = pickle.load(f)
                all_regret.append([0] + results_dict["cum_regret"][1:])

            if len(all_regret) == 0:
                continue

            all_regret = np.array(all_regret)

            mean_reg = all_regret.mean(axis=0)
            acc = (len(mean_reg) - mean_reg[-1]) * 100 / len(mean_reg)

            overall_acc.append(acc)
            dataset_acc[dataset][alg] = acc

        result[alg] = sum(overall_acc) / len(overall_acc) if len(overall_acc) > 0 else 0

    improvement = 0
    for dataset in datasets:
        obj_copy = deepcopy(dataset_acc[dataset])
        c3_acc = obj_copy.pop("c3")
        max_acc = np.max(list(obj_copy.values()))

        improvement += c3_acc - max_acc
    improvement /= len(datasets)
    print(f"Improvement of C3 over the best/next best algorithm, averaged over datasets is {improvement}")

    return result, dataset_acc


if __name__ == "__main__":
    VERBOSE = False

    base_dir = "results"
    separate_results(base_dir, ["shuttle", "magic", "cover", "mnist"],
                                ["C3", "linucb", "neuralucb", "lts", "neuralts", "squarecb"],
                                ["$C_3$", "LinUCB", "NeuralUCB", "LinTS", "NeuralTS", "SquareCB"],
                                "cumregret", (2, 2),
                                titles=["shuttle", "MagicTelescope", "covertype", "MNIST"])
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", "bandit_regret.png"), dpi=200)
