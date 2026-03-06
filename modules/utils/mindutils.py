import numpy as np
import torch

from torch.utils.data import Dataset


def get_context_sentence(history, news_dict, category_map):
    category_freq = [0.] * len(category_map)
    for hist in history:
        category = news_dict[hist]["vertical"]
        category_freq[category_map[category]] += 1

    return np.array(category_freq)


def get_categories(imp_list, news_dict, category_map):
    category_freq = [0.] * len(category_map)
    for imp, clicked in imp_list:
        category = news_dict[imp]["vertical"]
        category_freq[category_map[category]] += 1

    return np.array(category_freq)


def get_clicked_stats(imp_list, news_dict, category_map):
    category_freq = [0.] * len(category_map)
    for imp, clicked in imp_list:
        if clicked == 1:
            category = news_dict[imp]["vertical"]
            category_freq[category_map[category]] += 1

    return np.array(category_freq)


def augment_clickrate(category, prob_dict, news_dict):
    def _f(imp):
        imp_out = list()
        day, imp = imp.iloc[0], imp.iloc[1]
        prob = prob_dict[day]
        for news_id, click in imp:
            if news_dict[news_id]["vertical"] == category and click == 0:
                click = np.random.choice([0, 1], p=[1 - prob, prob])
            imp_out.append((news_id, click))
        return imp_out

    return _f


def diminish_clickrate(neg_category, prob_dict, news_dict):
    def _f(imp):
        imp_out = list()
        day, imp = imp.iloc[0], imp.iloc[1]
        prob = prob_dict[day]
        for news_id, click in imp:
            if news_dict[news_id]["vertical"] != neg_category and click == 1:
                click = np.random.choice([0, 1], p=[prob, 1 - prob])
            imp_out.append((news_id, click))
        return imp_out

    return _f


def indexer(i, imp_values, title_dict):
    ctx, imp = imp_values[i]
    all_arms, all_targets = zip(*[(title_dict[nid], target) for nid, target in imp])
    return (torch.Tensor(ctx).unsqueeze(dim=0),
            torch.stack(all_arms).unsqueeze(dim=0),
            torch.Tensor(all_targets).unsqueeze(dim=0))


class Indexer(Dataset):
    def __init__(self, imp_table, title_dict):
        super().__init__()
        self.imp_values = imp_table.values
        self.title_dict = title_dict

    def __len__(self):
        return len(self.imp_values)

    def __getitem__(self, item):
        return indexer(item, self.imp_values, self.title_dict)


class MINDStreamer:
    def __init__(self, impressions, title_dict):
        self.impressions = impressions.values
        self.title_dict = title_dict

        self.rng = None
        self.indices = None
        self.i = None
        self.X_out = None
        self.X_sep = None
        self.target = None

    def reset(self, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.indices = np.arange(len(self.impressions))
        self.rng.shuffle(self.indices)
        self.i = 0

        context, arms, self.target = self.preprocess_imp_twotower(self.impressions[self.i])
        self.X_sep = (context, arms)
        self.X_out, self.target = self.preprocess_imp(self.impressions[self.i])

        return self.X_out, self.X_sep

    def take_action(self, arm):
        reward = self.target[arm]
        best_reward = self.target.max()

        return reward, best_reward

    def next(self):
        self.i += 1
        if self.i >= len(self.impressions):
            return None, None

        context, arms, self.target = self.preprocess_imp_twotower(self.impressions[self.i])
        self.X_sep = (context, arms)
        self.X_out, self.target = self.preprocess_imp(self.impressions[self.i])

        return self.X_out, self.X_sep

    def preprocess_imp(self, imp):
        ctx, articles = imp

        X_out = list()
        y_out = list()

        for nid, target in articles:
            X_out.append(ctx.tolist() + self.title_dict[nid].tolist())
            y_out.append(float(target))

        return torch.Tensor(X_out), torch.Tensor(y_out)

    def preprocess_imp_twotower(self, imp):
        ctx, articles = imp
        all_arms, all_targets = zip(*[(self.title_dict[nid], target) for nid, target in articles])
        return (torch.Tensor(ctx).unsqueeze(dim=0),
                torch.stack(all_arms).unsqueeze(dim=0),
                torch.Tensor(all_targets))
