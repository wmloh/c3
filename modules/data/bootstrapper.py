import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class Bootstrapper(Dataset):
    def __init__(self, X_data, y_data, usage_ratio=1, base_ratio=0.6, seed=None):
        super().__init__()
        assert len(X_data) == len(y_data)

        rng = np.random.default_rng(seed=seed)

        if usage_ratio < 1:
            idx = np.arange(len(X_data))
            rng.shuffle(idx)
            idx = idx[:int(usage_ratio * len(X_data))]

            X_data = X_data[idx]
            y_data = y_data[idx]

        total_length = len(X_data)
        indices = np.zeros(total_length, dtype=bool)
        indices[rng.choice(total_length, size=int(base_ratio * total_length), replace=False)] = True

        self.X_base = X_data[indices]
        self.y_base = y_data[indices]

        self.X_boot = X_data[~indices]
        self.y_boot = y_data[~indices]

    def get_ref_data(self):
        return self.X_base, self.y_base

    def get_query_data(self):
        return self.X_boot, self.y_boot

    def __len__(self):
        return len(self.X_boot)

    def __getitem__(self, item):
        return self.X_boot[item].unsqueeze(dim=0), self.y_boot[item]


class MultiBootstrapper(Dataset):
    def __init__(self, X_list, y_list, usage_ratio=1, base_ratio=0.6, seed=None,
                 batch_size=16, shuffle=True):
        super().__init__()
        assert len(X_list) == len(y_list)
        assert len(X_list[0]) == len(y_list[0])

        self.block_size = len(X_list[0])
        self.list_len = len(X_list)
        self.total_length = self.block_size * len(X_list)

        self.boots = [Bootstrapper(X_data, y_data,
                                   usage_ratio=usage_ratio, base_ratio=base_ratio,
                                   seed=None if seed is None else (seed + i)) for
                      i, (X_data, y_data) in
                      enumerate(zip(X_list, y_list))]
        self.dls = [DataLoader(ds, batch_size=batch_size, shuffle=shuffle) for ds in self.boots]

    def __len__(self):
        return self.list_len

    def __getitem__(self, item):
        return self.dls[item], self.boots[item].get_ref_data()

    def get_dl(self):
        return DataLoader(self, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
