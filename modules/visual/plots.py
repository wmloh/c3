import numpy as np
import matplotlib.pyplot as plt


def plot_embeddings(emb_red, y_train, A_data, R_data, num_arms, save_path=None, figsize=(12, 5)):
    num_points = len(emb_red)
    plt.figure(figsize=figsize)
    plt.subplot(121)

    plt.scatter(*emb_red[y_train[:num_points].numpy().squeeze() == 0].T, s=5, label="Wrong", c="tab:red")
    plt.scatter(*emb_red[y_train[:num_points].numpy().squeeze() == 1].T, s=5, label="Right", c="tab:green")
    plt.title("Reward Distribution")
    plt.legend()

    plt.subplot(122)
    for i in range(num_arms):
        plt.scatter(*emb_red[A_data[:num_points] == i].T, s=5, label=str(i))
    plt.title("Action Distribution")
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=100)
        plt.close()


def plot_cum_regret(cum_regret_list, all_actions_list, titles, num_arms,
                    figsize=(10, 6), save_path=None):
    num_algos = len(cum_regret_list)
    assert num_algos == len(titles)

    plt.figure(figsize=figsize)
    plt.subplot(121)
    for cumreg, title in zip(cum_regret_list, titles):
        plt.plot(cumreg, label=title)
    plt.plot([0, len(cumreg)], [0, len(cumreg)], ":", label="Maximum cumulative regret")
    plt.ylabel("Cumulative regret")
    plt.xlabel("Steps")
    plt.legend()

    for i, (actions, title) in enumerate(zip(all_actions_list, titles)):
        plt.subplot(num_algos, 2, (i + 1) * 2)
        plt.hist(actions, bins=np.arange(num_arms + 1, dtype=int), align='left', rwidth=0.5)
        plt.title(f"Frequency of arms pulled ({title})")
    plt.xlabel("Action")
    plt.ylabel("Frequency")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
        plt.close()


def plot_buffer_embeddings(emb_red, model, num_arms, start_index=None, save_path=None, figsize=(12, 5)):
    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.scatter(*emb_red[model.y_buff[start_index:].numpy().squeeze() == 0].T, s=5, label="Wrong", c="tab:red")
    plt.scatter(*emb_red[model.y_buff[start_index:].numpy().squeeze() == 1].T, s=5, label="Right", c="tab:green")
    plt.title("Reward Distribution")
    plt.legend()

    plt.subplot(122)
    for i in range(num_arms):
        plt.scatter(*emb_red[model.X_buff[start_index:, -num_arms:].argmax(dim=-1).numpy() == i].T, s=5, label=str(i))
    plt.title("Action Distribution")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
        plt.close()
