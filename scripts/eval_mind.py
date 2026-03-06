import os
import time
import logging
import logging.config
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from modules.interface.args import parse_eval_mind
from modules.model.c3_paper import C3  # uses legacy C3 implementation (only for reproducibility; not recommended)
# from modules.model.c3 import C3  # new implementation (recommended for most cases)
from modules.utils.mindutils import MINDStreamer, get_context_sentence, get_categories, get_clicked_stats, augment_clickrate, diminish_clickrate, Indexer
from baselines.nonstationary.bayeslr import BayesLR
from baselines.nonstationary.twotower import TwoTower
from baselines.nonstationary.swlinucb import SWUCB
from baselines.nonstationary.dlinucb import DLinUCB
from baselines.nonstationary.gpforgetting import ContextualGPBandit
from baselines.nonstationary.onlinenucb import OnlineNeuralUCB
from baselines.nonstationary.onlinents import OnlineNeuralTS
from baselines.nonstationary.crb import DynamicFeatureCRB


SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "interactive")
LOG_FILE_NAME = f"slurm-{SLURM_JOB_ID}.log"

# obtaining arguments and configs
args, config, params = parse_eval_mind()

# ================ #
# GLOBAL CONSTANTS #
# ================ #

NAME_MAP = {
    "C3": r"$C_3$",
    "bayeslr": "BayesLR",
    "twotower": "TwoTower",
    "swlinucb": "SW-LinUCB",
    "dlinucb": "D-LinUCB",
    "gpforget": "Gaussian Process",
    "onlinenucb": "Online-NUCB",
    "onlinents": "Online-NTS",
    "crb": "CRB"
}

SEED = args.seed
EXP_DIR = args.exp_dir
ALGORITHM = args.algorithm
TRAIN_DATA_DIR = args.train_data_dir
VAL_DATA_DIR = args.val_data_dir
EMBEDDING_DIR = args.embedding_dir
CONFIG_PATH = args.config
MODEL_CONFIG_PATH = args.model_config

MAX_BEHAVIOUR = config["MAX_BEHAVIOUR"]
TRAIN_RATIO = config["TRAIN_RATIO"]
MAX_IMP = config["MAX_IMP"]
SPLIT_SIZE = config["SPLIT_SIZE"]
TEST_POINTS = config["TEST_POINTS"]
SAMPLE_RATIO = config["SAMPLE_RATIO"]
TRAIN_DAYS = config["TRAIN_DAYS"]
TEST_DAYS = config["TEST_DAYS"]
EMB_DIM = config["EMB_DIM"]
TARGET_CATEGORY = config["TARGET_CATEGORY"]
AUG_DICT = dict(config["AUG_DICT"])
DIM_DICT = dict(config["DIM_DICT"])

EMB_FILE_NAME = "embedding.npy"
NEWS_FILE_NAME = "news_id.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ #
# SCRIPT SETUP #
# ============ #

if "1" in os.environ.get("BATCHFLAG", {}):
    # when sbatch is used (non-interactive), flush logs to the results directory
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%d/%m %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.exp_dir, LOG_FILE_NAME),
        filemode="a"
    )
else:
    # when run interactively, flush logs to stdout
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%d/%m %H:%M:%S",
        level=logging.INFO
    )


def get_path(x): return os.path.join(EXP_DIR, str(SEED), x)


np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.info(f"Running {ALGORITHM} on MIND with seed {SEED}")

# ============================ #
# DATA LOADING & PREPROCESSING #
# ============================ #

# Loading data
logging.info(f"Loading news data")
news_train = pd.read_table(os.path.join(TRAIN_DATA_DIR, "news.tsv"),
                           names=['newid', 'vertical', 'subvertical', 'title',
                                  'abstract', 'url', 'entities in title', 'entities in abstract'])
news_val = pd.read_table(os.path.join(VAL_DATA_DIR, "news.tsv"),
                         names=['newid', 'vertical', 'subvertical', 'title',
                                'abstract', 'url', 'entities in title', 'entities in abstract'])
news = pd.concat([news_train, news_val], ignore_index=True)
news.drop_duplicates("newid", inplace=True, ignore_index=True)
news_dict = news.set_index("newid").to_dict("index")
category_map = {k: i for i, k in enumerate(sorted(news["vertical"].unique()))}

NUM_CAT = len(category_map)

logging.info(f"{NUM_CAT = } | {EMB_DIM = }")

del news_train
del news_val

if ALGORITHM == "C3":
    if params["init"]["layer_nums"][0] is None:
        params["init"]["layer_nums"][0] = NUM_CAT + EMB_DIM
elif ALGORITHM == "twotower":
    if params["init"]["user_encoder_layers"][0] is None:
        params["init"]["user_encoder_layers"][0] = NUM_CAT
    if params["init"]["arm_encoder_layers"][0] is None:
        params["init"]["arm_encoder_layers"][0] = EMB_DIM

logging.info(f"Loading behaviour data")

behaviour_train = pd.read_table(os.path.join(TRAIN_DATA_DIR, "behaviors.tsv"),
                                names=["impid", "userid", "time", "history", "impressions"])
behaviour_val = pd.read_table(os.path.join(VAL_DATA_DIR, "behaviors.tsv"),
                              names=["impid", "userid", "time", "history", "impressions"])
behaviour = pd.concat([behaviour_train, behaviour_val], ignore_index=True)

del behaviour_train, behaviour_val

behaviour = behaviour.sample(frac=SAMPLE_RATIO, random_state=SEED, axis=0, ignore_index=True)

behaviour["time"] = pd.to_datetime(behaviour["time"], format="%m/%d/%Y %I:%M:%S %p")
behaviour["day"] = behaviour["time"].apply(lambda x: x.day)
behaviour["month"] = behaviour["time"].apply(lambda x: x.month)

behaviour["impressions"] = behaviour["impressions"].apply(lambda x: [(y.split("-")[0], int(y.split("-")[1])) for y in x.split(" ")])
behaviour["impressions"] = behaviour[["day", "impressions"]].apply(augment_clickrate(TARGET_CATEGORY, AUG_DICT, news_dict), axis=1)
behaviour["impressions"] = behaviour[["day", "impressions"]].apply(diminish_clickrate(TARGET_CATEGORY, DIM_DICT, news_dict), axis=1)

behaviour["num_imp"] = behaviour["impressions"].apply(len)
behaviour = behaviour[behaviour["num_imp"] <= MAX_IMP]
behaviour = behaviour[behaviour["history"].apply(lambda x: isinstance(x, str)).values]

behaviour["imp_clicked_stats"] = behaviour["impressions"].apply(lambda x: get_clicked_stats(x, news_dict, category_map))
behaviour["imp_categories"] = behaviour["impressions"].apply(lambda x: get_categories(x, news_dict, category_map))

behaviour["history"] = behaviour["history"].apply(lambda x: x.split())
behaviour["context"] = behaviour["history"].apply(lambda x: get_context_sentence(x, news_dict, category_map))

behaviour = behaviour.sort_values("time", ascending=True)
behaviour.drop(columns=["history", "time", "impid", "userid"], inplace=True)

logging.info(f"Done loading all data")

# Train validation test split
train_df = behaviour[behaviour["day"].isin(TRAIN_DAYS)][["context", "impressions", "day"]]
train_df, val_df = train_test_split(train_df, train_size=TRAIN_RATIO, random_state=SEED)
imp_train = {d: train_df[train_df["day"] == d][["context", "impressions"]] for d in TRAIN_DAYS}
imp_val = {d: val_df[val_df["day"] == d][["context", "impressions"]] for d in TRAIN_DAYS}

test_df = behaviour[behaviour["day"].isin(TEST_DAYS)][["context", "impressions", "day"]]
test_df = test_df.iloc[np.linspace(0, len(test_df) - 1, num=TEST_POINTS, dtype=int)]

# Encoding title with BERT
if os.path.exists(os.path.join(EMBEDDING_DIR, EMB_FILE_NAME)) and os.path.exists(os.path.join(EMBEDDING_DIR, NEWS_FILE_NAME)):
    title_emb = torch.Tensor(np.load(os.path.join(EMBEDDING_DIR, EMB_FILE_NAME), allow_pickle=True)).float()
    news_id = np.load(os.path.join(EMBEDDING_DIR, NEWS_FILE_NAME), allow_pickle=True)

    logging.info(f"Loading BERT embeddings from {EMBEDDING_DIR}")
    title_dict = {k: v for k, v in zip(news_id, title_emb)}
else:
    bert_name = "bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert = BertModel.from_pretrained(bert_name)

    bert = bert.to(DEVICE)

    logging.info("Generating BERT embeddings")
    with torch.no_grad():
        tokens = tokenizer(news["title"].values.tolist(), add_special_tokens=True, padding=True, return_tensors="pt").to(DEVICE)
        title_emb_list = list()
        for idx in trange(0, len(news), SPLIT_SIZE):
            title_emb_list.append(bert(tokens["input_ids"][idx: idx + SPLIT_SIZE],
                                       attention_mask=tokens["attention_mask"][idx: idx + SPLIT_SIZE]).pooler_output.cpu())
        title_emb = torch.vstack(title_emb_list)

    # dimensionality reduction
    pca = PCA(n_components=EMB_DIM)
    title_emb = torch.Tensor(pca.fit_transform(title_emb.numpy())).float()

    np.save(os.path.join(EMBEDDING_DIR, EMB_FILE_NAME), title_emb)
    np.save(os.path.join(EMBEDDING_DIR, NEWS_FILE_NAME), news["newid"].values)
    logging.info(f"BERT embeddings and news IDs saved to {EMBEDDING_DIR}")

    title_dict = {k: v for k, v in zip(news["newid"].values, title_emb)}

    tokens = tokens.to("cpu")
    bert = bert.cpu()

# Packaging up data into Tensors
X_train_list = list()
y_train_list = list()

X_val_list = list()
y_val_list = list()

for d in tqdm(TRAIN_DAYS, total=len(TRAIN_DAYS)):

    X_train = list()
    y_train = list()

    X_val = list()
    y_val = list()

    # concatenate context and arm features
    for ctx, imp in imp_train[d].values:
        for nid, target in imp:
            X_train.append(np.hstack([ctx, title_dict[nid]]).tolist())
            y_train.append(float(target))

    for ctx, imp in imp_val[d].values:
        for nid, target in imp:
            X_val.append(np.hstack([ctx, title_dict[nid]]).tolist())
            y_val.append(float(target))

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).unsqueeze(dim=-1)

    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val).unsqueeze(dim=-1)

    logging.info(f"Day {d} sample size: {len(X_train):,}")

    X_train_list.append(X_train)
    y_train_list.append(y_train)

    X_val_list.append(X_val)
    y_val_list.append(y_val)

logging.info(f"{X_train_list[0].shape = } | {y_train_list[0].shape = }")

# ===================== #
# TRAINING & EVALUATION #
# ===================== #

cum_clicks = [0]
cum_regret = [0]
time_steps = list()

streamer = MINDStreamer(test_df[["context", "impressions"]], title_dict)

start_time = time.time()

if ALGORITHM == "C3":
    # Phi training
    logging.info(f"Training C3Model")
    model = C3(X_init=torch.vstack(X_train_list), y_init=torch.vstack(y_train_list), seed=SEED, **params["init"])
    logging.info(f"X_init.shape={model.X_init.shape}")

    model.fit_time_partitioned(X_train_list, y_train_list, DEVICE, X_val=X_val_list, y_val=y_val_list,
                               val_seed=421582319,
                               save_plot_name=get_path("loss_plot.png"), **params["fit"])
    torch.save(model, get_path("c3model_offline.pt"))

    # Generating resultant embedding plots
    logging.info(f"Generating embedding plot")

    with torch.no_grad():
        emb_red = TSNE(perplexity=35).fit_transform(model.project(X_train_list[-1][:1000]).numpy())
    np.save(get_path("embred_offline.npy"), emb_red)

    plt.figure(figsize=(8, 5))
    plt.scatter(*emb_red[y_train_list[-1][:1000].numpy().squeeze() == 0].T, s=5, label="Not clicked")
    plt.scatter(*emb_red[y_train_list[-1][:1000].numpy().squeeze() == 1].T, s=5, label="Clicked")
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_path("embedding_offline.png"))
    plt.close()

    logging.info(f"Starting online bandit evaluation [{int(time.time() - start_time)}] sec")
    model.clear_buffer(seed=SEED)
    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        action, mean, imp_w, *_ = model.infer_batch(X)
        reward, optimal_R = streamer.take_action(action)

        # update reference set
        model.store_buffer(X[action].view(1, -1),
                           reward.view(1, 1),
                           torch.Tensor(imp_w))

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        if e % 100 == 99:
            idx = e // 100
            if params["drop_schedule"] and idx < len(params["drop_schedule"]):
                model.pop_buffer(slice(params["drop_schedule"][idx]))
            elif params["drop_prob"]:
                model.pop_buffer(np.where(np.random.binomial(1, p=params["drop_prob"], size=len(model.X_buff)).astype(bool))[0].tolist())

        X, X_sep = streamer.next()

elif ALGORITHM == "twotower":
    ttmodel = TwoTower(**params["init"])

    ds = Indexer(train_df[["context", "impressions"]], title_dict)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=lambda x: list(zip(*x)))

    ds_val = Indexer(val_df[["context", "impressions"]], title_dict)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    logging.info(f"Training TwoTower")
    ttmodel.fit(dl, dl_val, "cpu", save_plot_name=get_path("loss_plot.png"))

    X, X_sep = streamer.reset(seed=SEED)
    for e in trange(len(test_df)):
        pred = ttmodel(*X_sep)
        action = pred.argmax().item()
        reward, optimal_R = streamer.take_action(action)

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

elif ALGORITHM == "swlinucb":
    logging.info(f"Training SWLinUCB")

    swucb = SWUCB(d=X_train_list[0].size(1), **params["init"])
    window_size = params["init"]["w"]
    num_samples = 0

    X_samples = list()
    y_samples = list()

    for X_train, y_train in zip(X_train_list[::-1], y_train_list[::-1]):
        for X, y in zip(X_train, y_train):
            X_samples.append(X.numpy())
            y_samples.append(y.item())
            num_samples += 1

            if num_samples == window_size:
                break
        else:
            continue
        break

    assert len(X_samples) == window_size

    swucb.update(X_samples, y_samples, batched=True)

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        X = X.numpy()

        action, selected_candidate = swucb.select_action(X)

        reward, optimal_R = streamer.take_action(action)

        swucb.update(selected_candidate, reward.item())

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

elif ALGORITHM == "dlinucb":
    logging.info(f"Training DLinUCB")

    dlinucb = DLinUCB(d=X_train_list[0].size(1), **params["init"])

    for X_train, y_train in zip(X_train_list, y_train_list):
        for X, y in zip(X_train, y_train):
            dlinucb.update(X.unsqueeze(dim=0).numpy(), y.item())

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        X = X.numpy()

        action, selected_candidate = dlinucb.select_action(X)

        reward, optimal_R = streamer.take_action(action)

        dlinucb.update(selected_candidate, reward.item())

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

elif ALGORITHM == "gpforget":
    logging.info(f"Training GPForget")

    X_buffer = torch.cat(X_train_list, dim=0)
    y_buffer = torch.cat(y_train_list, dim=0)

    selector = np.random.choice(len(y_buffer), size=int(len(y_buffer) * params["sampling"]["ratio"]), replace=False).tolist()

    X_buffer = X_buffer[selector]
    y_buffer = y_buffer[selector].squeeze()

    logging.info(f"{X_buffer.shape = }")

    gp = ContextualGPBandit(X_buffer, y_buffer, **params["init"])
    remove_factor = params["update"]["remove_factor"]

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        X = X.numpy()

        action, selected_candidate = gp.select_action(X)

        reward, optimal_R = streamer.take_action(action)

        gp.update(selected_candidate, reward.item(), remove_factor=remove_factor)

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

elif ALGORITHM == "onlinenucb":
    logging.info(f"Training OnlineNeuralUCB")

    nucb = OnlineNeuralUCB(dim=X_train_list[0].size(1), **params["init"])

    for X_train, y_train in zip(X_train_list, y_train_list):
        for X, y in zip(X_train, y_train):
            nucb.insert_data(X.unsqueeze(dim=0).numpy(), y.item(), trim_excess=False)
    nucb.update()

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        X = X.numpy()

        action, selected_candidate = nucb.select_action(X)

        reward, optimal_R = streamer.take_action(action)

        nucb.insert_data(selected_candidate.detach().numpy(), reward.item(), trim_excess=True)

        nucb.update()

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

elif ALGORITHM == "onlinents":
    logging.info(f"Training OnlineNeuralTS")

    nts = OnlineNeuralTS(dim=X_train_list[0].size(1), **params["init"])

    for X_train, y_train in zip(X_train_list, y_train_list):
        for X, y in zip(X_train, y_train):
            nts.insert_data(X.unsqueeze(dim=0).numpy(), y.item(), trim_excess=False)
    nts.update()

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        X = X.numpy()

        action, selected_candidate = nts.select_action(X)

        reward, optimal_R = streamer.take_action(action)

        nts.insert_data(selected_candidate.detach().numpy(), reward.item(), trim_excess=True)

        nts.update()

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

elif ALGORITHM == "crb":
    logging.info(f"Training CRB")

    crb = DynamicFeatureCRB(dim=X_train_list[0].size(1), **params["init"])

    for X_train, y_train in zip(X_train_list, y_train_list):
        for X, y in zip(X_train, y_train):
            crb.update(X.numpy(), y.item())

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        X = X.numpy()

        action, selected_candidate = crb.select_action(X)

        reward, optimal_R = streamer.take_action(action)

        crb.update(selected_candidate, reward.item())

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

else:  # Bayesian linear regression
    lr = BayesLR(NUM_CAT + EMB_DIM)
    logging.info(f"Training BayesLR")
    lr.update_posterior(torch.vstack(X_train_list).numpy(),
                        torch.vstack(y_train_list).numpy().reshape(-1))

    X, X_sep = streamer.reset(seed=SEED)

    for e in trange(len(test_df)):
        mean_lr, stderr_lr, _ = lr(X)
        action_lr = np.argmax(mean_lr + stderr_lr)

        reward_lr, optimal_R = streamer.take_action(action_lr)

        lr.update_posterior(X[[action_lr]].numpy(), reward_lr.numpy().reshape(-1))
        lr.replace_prior()

        cum_clicks.append(cum_clicks[-1] + optimal_R)
        cum_regret.append(cum_regret[-1] + (optimal_R - reward_lr))
        time_steps.append(time.time() - start_time)

        X, X_sep = streamer.next()

logging.info(f"Done online bandit evaluation [{int(time.time() - start_time)} sec]")

# ====================== #
# SAVING AND VISUALIZING #
# ====================== #

results_dict = {
    "cum_regret": cum_regret,
    "cum_clicks": cum_clicks,
    "time_steps": time_steps
}
with open(get_path("results_dict.pkl"), "wb") as f:
    pickle.dump(results_dict, f)

plt.figure()
plt.plot(cum_clicks, ":", label="Total possible clicks", c="tab:green")
plt.plot(cum_regret, label=NAME_MAP[ALGORITHM])
plt.ylabel("Cumulative regret")
plt.xlabel("Steps")
plt.legend()
plt.savefig(get_path("results.png"), dpi=100)
plt.close()

logging.info(f"Execution complete")
