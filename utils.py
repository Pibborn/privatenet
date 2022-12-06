import numpy as np
import pandas as pd
import crypten
import torch


def load_data(x_loc, y_loc):
    x = pd.read_csv(x_loc, delimiter=' ', header=None)
    y = pd.read_csv(y_loc, delimiter='\n', header=None)
    y = (y - y.min()) / (y.max() - y.min())
    return pd.concat([x, y], axis=1)


def split_parties(df, n_parties=15, random_state=0):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return np.array_split(df, n_parties)


def split_x_y(df):
    return df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()


def get_n_features(x):
    return len(x[0])


def convert_and_unsqueeze(y):
    return torch.unsqueeze(torch.Tensor(y), -1)

def encode_splits(splits):
    for i, split in enumerate(splits):
        x, y = split_x_y(split)
        x = crypten.cryptensor(x, src=i)
        y = convert_and_unsqueeze(y)
        y = crypten.cryptensor(y, src=i)
        splits[i] = (x, y)
    return splits

def combine_splits(splits):
    x = [split[0] for split in splits]
    y = [split[1] for split in splits]
    return crypten.cat(x, dim=0), crypten.cat(y, dim=0)
