import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.ncmapss import flatten_RUL


class CMAPSSDataset(Dataset):

    def __init__(self, path, n_DS="FD001", mode="train", units=None, concepts=["HPC", "Fan+HPC"], combined_concepts=False, binary_concepts=True, RUL="linear", include_healthy=True, window_size=1, stride=1, **kwargs):

        assert RUL in ("linear", "flat"), "RUL type must be 'linear' or 'flat"
        print(f"Using {RUL} RUL.")
        self.n_DS = n_DS
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        self.df_X = pd.read_csv(os.path.join(path, f"{mode}_{n_DS}.txt"), sep=" ", header=None, index_col=False,
            names=["unit", "cycle"] \
                # + [f"operational_setting_{i+1}" for i in range(3)] \
                + ["alt", "Mach", "TRA"]
                # + [f"sensor_{i+1}" for i in range(21)]
                + ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"])
        if self.mode == "test":
            df_target = pd.read_csv(os.path.join(path, f"RUL_{n_DS}.txt"), header=None, names=["RUL"])
            df_target.index = pd.Index(range(1, df_target.shape[0]+1), dtype="int")

        # remove constant features
        # self.df_X = self.df_X.drop(["Mach", "TRA", "T2", "epr", "Nf_dmd", "PCNfR_dmd"], axis=1)

        num_cycles = self.df_X.groupby("unit").size()

        if self.mode == "train":  # full trajectories until EOL
            RUL_by_cycle = self.df_X.groupby("unit").apply(lambda df: pd.DataFrame({
                "RUL": np.arange(num_cycles[df.iloc[0]["unit"]]-1, -1, step=-1)
            })).reset_index()
        else:  # for test, RUL at last trajectory cycle is given as target in df_Y
            RUL_by_cycle = self.df_X.groupby("unit").apply(lambda df: pd.DataFrame({
                "RUL": np.arange(num_cycles[df.iloc[0]["unit"]]+df_target["RUL"][df.iloc[0]["unit"]]-1, df_target["RUL"][df.iloc[0]["unit"]]-1, step=-1)
            })).reset_index()
        RUL_by_cycle["cycle"] = RUL_by_cycle["level_1"]

        # health state  TODO: use real change point detection
        RUL_by_cycle["hs"] = (RUL_by_cycle["RUL"] > 125).astype(int)

        if RUL == "flat":
            # Set RUL to flat constant value when hs == 1
            RUL_by_cycle["RUL"] = RUL_by_cycle.groupby("unit", group_keys=False).apply(flatten_RUL)["RUL"]

        self.df_Y = RUL_by_cycle["RUL"]
        self.df_X["hs"] = RUL_by_cycle["hs"]

        del RUL_by_cycle

        if not include_healthy:
            self.df_X = self.df_X.loc[self.df_X['hs'] == 0]
            self.df_Y = self.df_Y.loc[self.df_X['hs'] == 0]

        if units is not None:
            # select only unit data
            self.df_Y = self.df_Y.loc[self.df_X['unit'].isin(units)]
            self.df_X = self.df_X.loc[self.df_X['unit'].isin(units)]

        if combined_concepts:
            self.concepts = pd.DataFrame({
                "HPC": int(n_DS in ["FD001", "FD002"]) * (1 - self.df_X["hs"]),
                "Fan+HPC": int(n_DS in ["FD003", "FD004"]) * (1 - self.df_X["hs"])
            })[[c for c in concepts if c not in ["healthy"]]]
        else:
            self.concepts = pd.DataFrame({
                "HPC": (1 - self.df_X["hs"]),
                "Fan": int(n_DS in ["FD003", "FD004"]) * (1 - self.df_X["hs"])
            })[[c for c in concepts if c not in ["healthy"]]]

        if not binary_concepts:
            raise NotImplementedError
        print("Used concepts:", self.concepts.columns)
        self.concepts = self.concepts.astype(int)

        self.X = self.df_X.drop(["unit", "cycle", "hs"], axis=1)

        # Min-max scaling (here, FD001)
        X_mins = {
            "FD001": np.array([-8.70000e-03, -6.00000e-04,  1.00000e+02,  5.18670e+02,
            6.41210e+02,  1.57104e+03,  1.38225e+03,  1.46200e+01,
            2.16000e+01,  5.49850e+02,  2.38790e+03,  9.02173e+03,
            1.30000e+00,  4.68500e+01,  5.18690e+02,  2.38788e+03,
            8.09994e+03,  8.32490e+00,  3.00000e-02,  3.88000e+02,
            2.38800e+03,  1.00000e+02,  3.81400e+01,  2.28942e+01]),
            "FD003": np.array([-8.60000e-03, -6.00000e-04,  1.00000e+02,  5.18670e+02,
            6.40840e+02,  1.56430e+03,  1.37706e+03,  1.46200e+01,
            2.14500e+01,  5.49610e+02,  2.38690e+03,  9.01798e+03,
            1.29000e+00,  4.66900e+01,  5.17770e+02,  2.38693e+03,
            8.09968e+03,  8.15630e+00,  3.00000e-02,  3.88000e+02,
            2.38800e+03,  1.00000e+02,  3.81700e+01,  2.28726e+01])
        }
        X_maxs = {
            "FD001": np.array([8.70000e-03, 6.00000e-04, 1.00000e+02, 5.18670e+02, 6.44530e+02,
            1.61691e+03, 1.44149e+03, 1.46200e+01, 2.16100e+01, 5.56060e+02,
            2.38856e+03, 9.24459e+03, 1.30000e+00, 4.85300e+01, 5.23380e+02,
            2.38856e+03, 8.29372e+03, 8.58480e+00, 3.00000e-02, 4.00000e+02,
            2.38800e+03, 1.00000e+02, 3.94300e+01, 2.36184e+01]),
            "FD003": np.array([8.60000e-03, 7.00000e-04, 1.00000e+02, 5.18670e+02, 6.45110e+02,
            1.61539e+03, 1.44116e+03, 1.46200e+01, 2.16100e+01, 5.70490e+02,
            2.38860e+03, 9.23435e+03, 1.32000e+00, 4.84400e+01, 5.37400e+02,
            2.38861e+03, 8.29055e+03, 8.57050e+00, 3.00000e-02, 3.99000e+02,
            2.38800e+03, 1.00000e+02, 3.98500e+01, 2.39505e+01])
        }
        X_min = np.minimum(X_mins["FD001"], X_mins["FD003"])
        X_max = np.maximum(X_maxs["FD001"], X_maxs["FD003"])
        #     X_min = np.array([-8.70000e-03,  6.41210e+02,  1.57104e+03,
        #     1.38225e+03,  1.46200e+01,  2.16000e+01,  5.49850e+02,
        #     2.38790e+03,  9.02173e+03,  4.68500e+01,  5.18690e+02,
        #     2.38788e+03,  8.09994e+03,  8.32490e+00,  3.00000e-02,
        #     3.88000e+02,  3.81400e+01,  2.28942e+01])
        #     X_max = np.array([8.70000e-03, 6.44530e+02, 1.61691e+03, 1.44149e+03,
        #    1.46200e+01, 2.16100e+01, 5.56060e+02, 2.38856e+03, 9.24459e+03,
        #    4.85300e+01, 5.23380e+02, 2.38856e+03, 8.29372e+03, 8.58480e+00,
        #    3.00000e-02, 4.00000e+02, 3.94300e+01, 2.36184e+01])
        X_std = (self.X - X_min) / (X_max - X_min + 1e-12)
        f_min, f_max = -1, 1
        self.X = X_std * (f_max - f_min) + f_min
        self.df_Y /= 100.  # max in FD001 is 362

        print(self.X.shape)

    def __len__(self):
        return self.X.shape[0] // self.stride #Floor total length divided by stride

    def __getitem__(self, i):
        #Sequence is entirely within the data
        if (i * self.stride >= self.window_size - 1):

            unit = self.df_X["unit"].iloc[i * self.stride - self.window_size + 1:i * self.stride + 1]  #Unit vector for desired sequence

            cond = np.where(unit != unit.iloc[0])[0] #Index of first different value

            if cond.size == 0: #If there is no index with different value
                i_start = i * self.stride - self.window_size + 1
                x = self.X.iloc[i_start:i * self.stride + 1, :].values.T
            else:
                counter = cond[0] #Find first index of switch
                padding = self.X.iloc[i * self.stride -(self.window_size - counter) + 1].values.reshape(-1, 1).repeat(counter, 1)
                x = self.X.iloc[i * self.stride - (self.window_size - counter) + 1:i * self.stride + 1, :].values.T
                x = np.concatenate((padding, x), 1)

            #Beginning of Sequence (backward filling)
        else:
            padding = self.X.iloc[0].values.reshape(-1, 1).repeat(self.window_size - i * self.stride - 1, 1)
            x = self.X.iloc[0:i * self.stride + 1, :].values.T
            x = np.concatenate((padding, x), 1)

        x = x.ravel()

        return torch.Tensor(x).squeeze(), torch.Tensor([self.df_Y.iloc[i * self.stride]]).squeeze(), torch.Tensor(self.concepts.iloc[i * self.stride].to_numpy())
