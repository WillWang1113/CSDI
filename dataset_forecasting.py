import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler


class Forecasting_Dataset(Dataset):
    def __init__(
        self,
        datatype,
        seq_len,
        pred_len,
        data_dir,
        target="OT",
        features="S",
        scale=True,
        mode="train",
    ):
        self.history_length = seq_len
        self.pred_length = pred_len
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[mode]

        df_raw = pd.read_csv(data_dir)
        cols = list(df_raw.columns)
        cols.remove(target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        if datatype.__contains__("etth"):
            border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]

        elif datatype.__contains__("ettm"):
            border1s = [
                0,
                12 * 30 * 24 * 4 - seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]

        else:
            border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        start = border1s[self.set_type]
        end = border2s[self.set_type]

        if features == "M" or features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif features == "S":
            df_data = df_raw[[target]]
        self.target_dim = len(df_data.columns)
            

        if scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.main_data = data[start:end]
        self.mask_data = np.ones_like(self.main_data)

        # if datatype == "electricity":
        #     datafolder = "./data/electricity_nips"
        #     self.test_length = 24 * 7
        #     self.valid_length = 24 * 5

        self.seq_length = self.history_length + self.pred_length

        # paths = datafolder + "/data.pkl"
        # # shape: (T x N)
        # # mask_data is usually filled by 1
        # with open(paths, "rb") as f:
        #     self.main_data, self.mask_data = pickle.load(f)
        # paths = datafolder + "/meanstd.pkl"
        # with open(paths, "rb") as f:
        #     self.mean_data, self.std_data = pickle.load(f)

        # self.main_data = (self.main_data - self.mean_data) / self.std_data

        # total_length = len(self.main_data)

    def __getitem__(self, orgindex):
        target_mask = self.mask_data[orgindex : orgindex + self.seq_length].copy()
        target_mask[-self.pred_length :] = 0.0  # pred mask for test pattern strategy
        s = {
            "observed_data": self.main_data[orgindex : orgindex + self.seq_length],
            "observed_mask": self.mask_data[orgindex : orgindex + self.seq_length],
            "gt_mask": target_mask,
            "timepoints": np.arange(self.seq_length) * 1.0,
            "feature_id": np.arange(self.main_data.shape[1]) * 1.0,
        }

        return s

    def __len__(self):
        return len(self.main_data) - self.seq_length + 1


def get_dataloader(
    datatype,
    device,
    seq_len,
    pred_len,
    data_dir,
    target="OT",
    features="S",
    scale=True,
    batch_size=8,
):
    dataset = Forecasting_Dataset(
        datatype,
        seq_len,
        pred_len,
        data_dir,
        target=target,
        features=features,
        scale=scale,
        mode="train",
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = Forecasting_Dataset(
        datatype,
        seq_len,
        pred_len,
        data_dir,
        target=target,
        features=features,
        scale=scale,
        mode="val",
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = Forecasting_Dataset(
        datatype,
        seq_len,
        pred_len,
        data_dir,
        target=target,
        features=features,
        scale=scale,
        mode="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    scaler = 1.0
    mean_scaler = 0.0
    target_dim = dataset.target_dim

    return train_loader, valid_loader, test_loader, scaler, mean_scaler, target_dim
