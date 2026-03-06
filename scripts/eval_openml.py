import os
import time
import logging
import logging.config
import pickle
import numpy as np
import torch

from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from striatum.bandit import LinUCB, LinThompSamp
from striatum.storage import MemoryHistoryStorage, MemoryModelStorage, MemoryActionStorage, Action
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from coba.learners import VowpalSquarecbLearner

from modules.interface.args import parse_eval_openml
from modules.data.dataconverter import DataConverter
from modules.model.c3_paper import C3  # uses legacy C3 implementation (only for reproducibility; not recommended)
# from modules.model.c3 import C3  # new implementation (recommended for most cases)
from modules.visual.plots import plot_embeddings, plot_cum_regret
from baselines.stationary.neuralucb import NeuralUCBDiag
from baselines.stationary.neuralts import NeuralTS
from baselines.stationary.neurallinear import NeuralLinear


SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "interactive")
LOG_FILE_NAME = f"slurm-{SLURM_JOB_ID}.log"

# obtaining arguments and configs
args, config = parse_eval_openml()

# ================ #
# GLOBAL CONSTANTS #
# ================ #

SEED = args.seed
EXP_DIR = args.exp_dir
ALGORITHM = args.algorithm
params = config["params"]
VERSION = config["VERSION"]
DATASET = config["DATASET"]
NUM_TEST_STEPS = config["NUM_TEST_STEPS"]
MAX_DATASET_SIZE = config["MAX_DATASET_SIZE"]
TRAIN_RATIO = config["TRAIN_RATIO"]
VAL_RATIO = config["VAL_RATIO"]
REWARD_PROP = config["REWARD_PROP"]
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


def get_path(x):
    return os.path.join(EXP_DIR, str(SEED), x)


np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.info(f"Running {ALGORITHM} on {DATASET} (v{VERSION}) with seed {SEED}")

# ============================ #
# DATA LOADING & PREPROCESSING #
# ============================ #

data_bunch = fetch_openml(DATASET, version=VERSION, parser="auto")

NUM_ARMS = len(set(data_bunch["target"]))
CTX_SIZE = data_bunch["data"].shape[1]

if params["init"]["layer_nums"][0] is None:
    params["init"]["layer_nums"][0] = CTX_SIZE + NUM_ARMS

# Data preprocessing
X = data_bunch["data"].values.astype(float)
y = OrdinalEncoder(dtype=int).fit_transform(data_bunch["target"].values.reshape(-1, 1)).astype(float)

if DATASET == "mnist_784":
    X = X / 255.  # map pixels to [0, 1]

X, y = shuffle(X, y, random_state=SEED)
X = X[:MAX_DATASET_SIZE]
y = y[:MAX_DATASET_SIZE]

# Train test splits and data converters
X, X_val, y, y_val = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=VAL_RATIO, random_state=SEED * 2)

dc = DataConverter(X, y, NUM_ARMS)
C_data, A_data, R_data, true_label = dc.get_pretraining_split(proportion=1,
                                                              reward_prop=REWARD_PROP,
                                                              seed=SEED)

dc_val = DataConverter(X_val, y_val, NUM_ARMS)
C_val_data, A_val_data, R_val_data, _ = dc_val.get_pretraining_split(proportion=1,
                                                                     reward_prop=REWARD_PROP,
                                                                     seed=SEED)

dc_test = DataConverter(X_test, y_test, NUM_ARMS)

X_train = torch.Tensor(np.hstack([C_data, np.eye(NUM_ARMS)[A_data]]))
y_train = torch.Tensor(R_data.reshape(-1, 1))
X_val = torch.Tensor(np.hstack([C_val_data, np.eye(NUM_ARMS)[A_val_data]]))
y_val = torch.Tensor(R_val_data.reshape(-1, 1))

logging.info(f"{X_train.shape = } | {X_val.shape = }")

# ===================== #
# TRAINING & EVALUATION #
# ===================== #

all_actions = list()
cum_regret = [0]
time_steps = list()

freq_eval = len(C_data) // len(dc_val)
start_time = time.time()

if ALGORITHM == "C3":
    # Phi training
    logging.info(f"Training C3Model")
    layer_nums = params["init"]["layer_nums"]

    model = C3(seed=SEED, **params["init"])

    model.fit(X_train, y_train, DEVICE, X_val=X_val, y_val=y_val,
              plot=True, save_plot_name=get_path("loss_plot.png"), **params["fit"])

    logging.info(f"Getting training regrets")
    model.clear_buffer(seed=SEED)
    C, C_arms = dc_val.reset(seed=SEED)

    for i in trange(len(dc_val)):
        action, mean, w_new, *_ = model.infer_batch(torch.Tensor(C_arms))
        reward, regret = dc_val.take_action(action)

        model.store_buffer(torch.Tensor(C_arms[A_val_data[i]]).view(1, -1),
                           torch.Tensor([R_val_data[i]]).view(1, 1),
                           torch.Tensor(w_new))
        model.store_buffer(torch.Tensor(X_train[i * freq_eval:(i + 1) * freq_eval]),
                           torch.Tensor(y_train[i * freq_eval:(i + 1) * freq_eval]))

        all_actions.append(action)
        cum_regret.append(cum_regret[-1] + regret)
        time_steps.append(time.time() - start_time)

        if i != len(dc_val) - 1:
            C, C_arms = dc_val.next()
        if i == NUM_TEST_STEPS:
            break

    model.X_init = X_train
    model.y_init = y_train

    # Generating resultant embedding plots
    logging.info(f"Generating embedding plot")

    with torch.no_grad():
        emb_red = TSNE(perplexity=35, random_state=SEED).fit_transform(model.project(X_train[:1000]).numpy())
    np.save(get_path("embred_offline.npy"), emb_red)
    plot_embeddings(emb_red, y_train, A_data, R_data, NUM_ARMS, save_path=get_path("embedding_offline.png"))

elif ALGORITHM == "linucb":
    linucb = LinUCB(MemoryHistoryStorage(), MemoryModelStorage(), MemoryActionStorage(), context_dimension=CTX_SIZE, alpha=1.96)
    linucb.add_action([Action() for _ in range(NUM_ARMS)])

    C, C_arms = dc_val.reset(seed=SEED)

    logging.info(f"Pretraining LinUCB")
    for i in range(len(C_data)):
        history_id, action_obj = linucb.get_action({j: C_data[i] for j in range(NUM_ARMS)})
        linucb.reward(history_id, {A_data[i]: R_data[i]})
        if i % 2000 == 0:
            logging.info(f"\tStep {i} / {len(C_data)}")

        if i % freq_eval == 0:
            j = i // freq_eval
            if j >= len(dc_val):
                continue
            history_id, action_obj = linucb.get_action({j: C for j in range(NUM_ARMS)})
            action = action_obj.action.id
            reward, regret = dc_val.take_action(action)

            linucb.reward(history_id, {A_val_data[j]: R_val_data[j]})

            all_actions.append(action)
            cum_regret.append(cum_regret[-1] + regret)
            time_steps.append(time.time() - start_time)

            if j != len(dc_val) - 1:
                C, C_arms = dc_val.next()

            if j == NUM_TEST_STEPS:
                break

elif ALGORITHM == "lts":
    lints = LinThompSamp(MemoryHistoryStorage(), MemoryModelStorage(), MemoryActionStorage(),
                         epsilon=1 / np.log(len(X_train) + len(X_test)),
                         context_dimension=CTX_SIZE, random_state=SEED)
    lints.add_action([Action() for _ in range(NUM_ARMS)])

    C, C_arms = dc_val.reset(seed=SEED)

    logging.info(f"Pretraining LinTS")
    for i in range(len(C_data)):
        history_id, action_obj = lints.get_action({j: C_data[i] for j in range(NUM_ARMS)})
        lints.reward(history_id, {A_data[i]: R_data[i]})
        if i % 2000 == 0:
            logging.info(f"\tStep {i} / {len(C_data)}")

        if i % freq_eval == 0:
            j = i // freq_eval
            if j >= len(dc_val):
                continue
            history_id, action_obj = lints.get_action({j: C for j in range(NUM_ARMS)})
            action = action_obj.action.id
            reward, regret = dc_val.take_action(action)

            lints.reward(history_id, {A_val_data[j]: R_val_data[j]})

            all_actions.append(action)
            cum_regret.append(cum_regret[-1] + regret)
            time_steps.append(time.time() - start_time)

            if j != len(dc_val) - 1:
                C, C_arms = dc_val.next()
            if j == NUM_TEST_STEPS:
                break

elif ALGORITHM == "squarecb":
    gamma = 10
    ALL_ACTIONS = list(range(NUM_ARMS))

    oracle = VowpalSquarecbLearner(gamma_scale=gamma)

    C, C_arms = dc_val.reset(seed=SEED)

    logging.info(f"Pretraining LinTS")
    for i in range(len(C_data)):
        action, prob, _ = oracle.predict(C_data[i].tolist(), [int(A_data[i])])
        oracle.learn(C_data[i].tolist(), action, R_data[i], prob)

        if i % freq_eval == 0:
            j = i // freq_eval
            if j >= len(dc_val):
                continue
            action, prob, _ = oracle.predict(C.tolist(), ALL_ACTIONS)
            reward, regret = dc_val.take_action(action)
            oracle.learn(C.tolist(), action, reward, prob)

            all_actions.append(action)
            cum_regret.append(cum_regret[-1] + regret)
            time_steps.append(time.time() - start_time)

            if j != len(dc_val) - 1:
                C, C_arms = dc_val.next()
            if j == NUM_TEST_STEPS:
                break

elif ALGORITHM == "neurallinear":
    nl = NeuralLinear(input_dim=CTX_SIZE + NUM_ARMS, embed_dim=NUM_ARMS)
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    nl.fit(train_dl, num_epochs=100)

    C, C_arms = dc_val.reset(seed=SEED)

    for i in trange(len(dc_val)):
        action = nl.pick_action(C_arms)
        reward, regret = dc_val.take_action(action)

        nl.update(C_arms[[action]], np.array([reward]))

        all_actions.append(action)
        cum_regret.append(cum_regret[-1] + regret)
        time_steps.append(time.time() - start_time)

        if i != len(dc_val) - 1:
            C, C_arms = dc_val.next()
        if i == NUM_TEST_STEPS:
            break

elif ALGORITHM == "neuralts":
    neuralts = NeuralTS(dim=CTX_SIZE * NUM_ARMS, nu=0.00001, lamdba=0.00001)

    C_val, C_arms_val = dc_val.reset(seed=SEED)

    logging.info(f"Pretraining NeuralTS")
    for i, (C, A, R) in enumerate(zip(C_data, A_data, R_data)):
        C_prime = NeuralTS.convert_data(C, NUM_ARMS)
        neuralts.force_select(C_prime, A)
        neuralts.train(C_prime[A], R)
        if i % 2000 == 0:
            logging.info(f"\tStep {i} / {len(C_data)}")

        if i % freq_eval == 0:
            j = i // freq_eval
            if j >= len(dc_val):
                continue
            C_prime_val = NeuralTS.convert_data(C_val, dc_val.num_arms)
            action, *_ = neuralts.select(C_prime_val)
            reward, regret = dc_val.take_action(action)

            neuralts.train(C_prime[A_val_data[j]], R_val_data[j])

            all_actions.append(action)
            cum_regret.append(cum_regret[-1] + regret)
            time_steps.append(time.time() - start_time)

            if j != len(dc_val) - 1:
                C_val, C_arms_val = dc_val.next()
            if j == NUM_TEST_STEPS:
                break

elif ALGORITHM == "neuralucb":
    neuralucb = NeuralUCBDiag(CTX_SIZE * NUM_ARMS, nu=0.00001, lamdba=0.00001)

    C_val, C_arms_val = dc_val.reset(seed=SEED)

    logging.info(f"Pretraining NeuralUCB [{int(time.time() - start_time)} sec]")
    for i, (C, A, R) in enumerate(zip(C_data, A_data, R_data)):
        C_prime = NeuralUCBDiag.convert_data(C, NUM_ARMS)
        neuralucb.force_select(C_prime, A)
        neuralucb.train(C_prime[A], R)
        if i % 2000 == 0:
            logging.info(f"\tStep {i} / {len(C_data)}")

        if i % freq_eval == 0:
            j = i // freq_eval
            if j >= len(dc_val):
                continue
            C_prime_val = NeuralUCBDiag.convert_data(C_val, dc_val.num_arms)
            action, *_ = neuralucb.select(C_prime_val)
            reward, regret = dc_val.take_action(action)

            neuralucb.train(C_prime[A_val_data[j]], R_val_data[j])

            all_actions.append(action)
            cum_regret.append(cum_regret[-1] + regret)
            time_steps.append(time.time() - start_time)

            if j != len(dc_val) - 1:
                C_val, C_arms_val = dc_val.next()
            if j == NUM_TEST_STEPS:
                break

else:
    raise ValueError(f"{ALGORITHM} is not a valid algorithm.")

logging.info(f"Done online bandit evaluation [{int(time.time() - start_time)} sec]")

# ====================== #
# SAVING AND VISUALIZING #
# ====================== #

results_dict = {
    "cum_regret": cum_regret,
    "all_actions": all_actions,
    "time_steps": time_steps
}

with open(get_path("results_dict.pkl"), "wb") as f:
    pickle.dump(results_dict, f)

plot_cum_regret([cum_regret], [all_actions], [ALGORITHM], NUM_ARMS,
                save_path=get_path("results.png"))

logging.info(f"Execution complete")
