import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class my_dataset(Dataset):
    def __init__(self, solutions, win) -> None:
        super().__init__()
        self.y = [torch.flatten(torch.from_numpy(sol.y)) for sol in solutions]
        self.t = [torch.flatten(torch.from_numpy(sol.t)) for sol in solutions]
        self.win = win

    def __len__(self):
        return sum([torch.numel(y) - self.win + 1 for y in self.y])

    def __getitem__(self, idx):
        len_list = [torch.numel(seq) - self.win + 1 for seq in self.y]
        cum_len = np.cumsum(len_list)
        series_index = np.count_nonzero(cum_len <= idx)
        if series_index > 0:
            idx = idx - cum_len[series_index - 1]

        return (
            self.t[series_index][idx : idx + self.win],
            self.y[series_index][idx].view(1),
            self.y[series_index][idx : idx + self.win],
        )


class approx_dynamics(nn.Module):
    def __init__(self, architecture: list = [100, 100]):
        super().__init__()
        new_arc = [1, *architecture, 1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features=in_f, out_features=out_f, dtype=torch.float64)
                for (in_f, out_f) in zip(new_arc[0:-1], new_arc[1:])
            ]
        )

    def forward(self, t, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)
