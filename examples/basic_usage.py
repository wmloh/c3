import matplotlib.pyplot as plt
import torch

from modules.model.c3 import C3

# =============== #
# CONFIGURE THIS! #
# =============== #

SEED = 0
D = 32  # feature dim
K = 4  # (fixed) number of arms
EMBEDDING_SIZE = 3

TRAIN_SIZE = 500
VAL_SIZE = 100
INIT_SIZE = 5

# ================== #
# DEFINING "DATASET" #
# ================== #

torch.manual_seed(SEED)


def get_reward(x):
    return torch.bernoulli(torch.sigmoid(x.sum(dim=-1, keepdim=True) * 5))


# creating dataset
X_0 = torch.randn(INIT_SIZE, D)  # (5, 32)
y_0 = get_reward(X_0)  # (5, 1)

X_train = torch.randn(TRAIN_SIZE, D)  # (500, 32)
y_train = get_reward(X_train)  # (500, 1)
X_val = torch.randn(VAL_SIZE, D)  # (500, 32)
y_val = get_reward(X_val)  # (500, 1)

# ======================== #
# C3 TRAINING & EVALUATION #
# ======================== #

# initializing C3
model = C3(
    layer_nums=[D, 8, EMBEDDING_SIZE],
    X_init=X_0,
    y_init=y_0,
    sigma=0.5,
    weight_factor=5.
)

# training phi (embedding model) of C3
model.fit(X_train, y_train, "cpu", X_val=X_val, y_val=y_val,
          loss_coef={"bce": 1., "ece": 5},
          epochs=20,
          lr=1e-3,
          usage_ratio=0.5,
          base_ratio=0.2,
          val_seed=SEED,
          plot=False
          )

# prepares for online learning / bandit evaluation
model.clear_buffer(seed=SEED)

# samples arm with Thompson sampling
X_stream = torch.randn(K, D)  # simulates one step with K-arms
action, mean, w_new, stddev, _ = model.infer_batch(X_stream)

# updates C3 with the chosen arm and new importance weights
reward = get_reward(X_stream[[action]])  # gets reward of the chosen arm
model.store_buffer(X_stream[[action]], reward, torch.tensor(w_new))

# ======================= #
# EMBEDDING VISUALIZATION #
# ======================= #

with torch.no_grad():
    # embed validation queries
    X_embed = model.project(X_val)
    X_buff = model.X_buff

    if X_embed.size(1) != 2:
        # reduce dimensionality if needed
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2).fit(model.project(X_train))
        X_embed = pca.transform(X_embed)
        X_buff = pca.transform(X_buff)

# get predicted reward conditioning on X_0 and y_0
_, pred_reward, _, stddev, eta = model.infer_batch(X_val)

# visualize query and reference embeddings with the predicted rewards
plt.figure(figsize=(9, 7.5))
plt.subplot(2, 2, 1)
sc = plt.scatter(*X_embed.T, c=pred_reward, cmap="coolwarm", label="Query", vmin=0., vmax=1.)
plt.scatter(*X_buff.T, c=model.y_buff, cmap="coolwarm", marker="*", s=100, label="Reference")
cbar = plt.colorbar(sc)
cbar.set_label("Predicted reward")
plt.xticks([])
plt.yticks([])
plt.xlabel("Embedding dimension 1")
plt.ylabel("Embedding dimension 2")
plt.title("IWKR reward predictions")
plt.legend()

# visualize query embeddings with the true rewards
plt.subplot(2, 2, 2)
sc = plt.scatter(*X_embed.T, c=y_val, cmap="coolwarm", alpha=0.5, edgecolors="none", label="Query")
plt.scatter(*X_buff.T, c=model.y_buff, cmap="coolwarm", marker="*", s=100, label="Reference")
cbar = plt.colorbar(sc)
cbar.set_label("True reward")
plt.xticks([])
plt.yticks([])
plt.xlabel("Embedding dimension 1")
plt.ylabel("Embedding dimension 2")
plt.title("Ground truth reward")
plt.legend()

plt.subplot(2, 2, 3)
sc = plt.scatter(*X_embed.T, c=stddev, cmap="summer", label="Query")
plt.scatter(*X_buff.T, c=model.y_buff, cmap="coolwarm", marker="*", s=100, label="Reference")
cbar = plt.colorbar(sc)
cbar.set_label("Standard deviation")
plt.xticks([])
plt.yticks([])
plt.xlabel("Embedding dimension 1")
plt.ylabel("Embedding dimension 2")
plt.title("Standard deviation of Beta posterior")
plt.legend()

plt.subplot(2, 2, 4)
sc = plt.scatter(*X_embed.T, c=eta, cmap="summer", label="Query")
plt.scatter(*X_buff.T, c=model.y_buff, cmap="coolwarm", marker="*", s=100, label="Reference")
cbar = plt.colorbar(sc)
cbar.set_label(r"$\eta(s)$")
plt.xticks([])
plt.yticks([])
plt.xlabel("Embedding dimension 1")
plt.ylabel("Embedding dimension 2")
plt.title(r"Kernel density estimates $\eta(s)$")
plt.legend()

plt.tight_layout()
plt.savefig("sample_embeddings.png", dpi=200)
