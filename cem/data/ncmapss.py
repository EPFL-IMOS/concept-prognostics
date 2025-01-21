import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from itertools import combinations


def subsampling(df, subsampling_rate):
    """
    reduce computational cost by subsampling the data
    :param df: pd.pd.DataFrame, subsampling data
    :param subsampling_rate: int, subsampling rate, reduce size to 1/subsampling_rate
    """
    return df[::subsampling_rate]


def binarize_concept(x):
    return x < -0.0015


def scale_concept(x):
    return np.clip(x / -0.035, 0, 1)


def flatten_RUL(df):
    hs_change = df[df["hs"].diff() == -1]
    if len(hs_change) > 0:
        df["RUL"].loc[df["cycle"] < hs_change.iloc[0]["cycle"]] = hs_change.iloc[0]["RUL"]
    return df


class NCMAPSSDataset(Dataset):

    def __init__(self, path, n_DS="02", mode="train", cruise=False, units=None, subsampling_rate=1, concepts=["LPT", "HPT"], combined_concepts=False, binary_concepts=True, RUL="linear", include_healthy=True, window_size=50, stride=1, scaling="legacy", **kwargs):

        assert RUL in ("linear", "flat"), "RUL type must be 'linear' or 'flat"
        print(f"Using {RUL} RUL.")
        self.n_DS = n_DS
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        filename = os.path.join(path, f"N-CMAPSS_DS{n_DS}.h5")
        with h5py.File(filename, 'r') as hdf:
            if 'train' in mode or 'val' in mode:
                # Development set
                W = np.array(hdf.get('W_dev'))             # W
                X_s = np.array(hdf.get('X_s_dev'))         # X_s
                X_v = np.array(hdf.get('X_v_dev'))         # X_v
                T = np.array(hdf.get('T_dev'))             # T
                Y = np.array(hdf.get('Y_dev'))             # RUL
                A = np.array(hdf.get('A_dev'))             # Auxiliary
            elif 'test' in mode:
                # Test set
                W = np.array(hdf.get('W_test'))           # W
                X_s = np.array(hdf.get('X_s_test'))       # X_s
                X_v = np.array(hdf.get('X_v_test'))       # X_v
                T = np.array(hdf.get('T_test'))           # T
                Y = np.array(hdf.get('Y_test'))           # RUL
                A = np.array(hdf.get('A_test'))           # Auxiliary

            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))
            X_v_var = np.array(hdf.get('X_v_var'))
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype='U20'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))
            X_v_var = list(np.array(X_v_var, dtype='U20'))
            T_var = list(np.array(T_var, dtype='U20'))
            A_var = list(np.array(A_var, dtype='U20'))

        #TODO: concat sensor data if virtual senssors should be considered
        # if sensors == 'all':
        #     X = np.concatenate((X_s, X_v), axis=-1)
        # else:
        #     X = X_s

        # aux data: 'unit', 'cycle', 'Fc', 'hs'
        self.df_A = pd.DataFrame(data=A, columns=A_var).astype(int)
        self.units = list(np.unique(self.df_A['unit']))

        # operating conditions: 'alt', 'Mach', 'TRA', 'T2'
        self.df_W = pd.DataFrame(data=W, columns=W_var)

        # degradation resp. concepts
        self.df_T = pd.DataFrame(data=T, columns=T_var)

        # sensor data
        self.df_X = pd.DataFrame(data=X_s, columns=X_s_var)

        # RUL
        self.df_Y = pd.DataFrame(data=Y, columns=['RUL'])

        del A, W, T, X_s, Y

        if RUL == "flat":
            # Set RUL to flat constant value when hs == 1
            df_all = pd.concat((self.df_A, self.df_Y), axis=1)
            self.df_Y["RUL"] = df_all.groupby("unit", group_keys=False).apply(flatten_RUL)["RUL"]
            del df_all

        if not include_healthy:
            self.df_W = self.df_W.loc[self.df_A['hs'] == 0]
            self.df_T = self.df_T.loc[self.df_A['hs'] == 0]
            self.df_X = self.df_X.loc[self.df_A['hs'] == 0]
            self.df_Y = self.df_Y.loc[self.df_A['hs'] == 0]
            self.df_A = self.df_A.loc[self.df_A['hs'] == 0]

        if units is not None:
            # select only unit data
            self.df_W = self.df_W.loc[self.df_A['unit'].isin(units)]
            self.df_T = self.df_T.loc[self.df_A['unit'].isin(units)]
            self.df_X = self.df_X.loc[self.df_A['unit'].isin(units)]
            self.df_Y = self.df_Y.loc[self.df_A['unit'].isin(units)]
            self.df_A = self.df_A.loc[self.df_A['unit'].isin(units)]

        if subsampling_rate > 1:
            # subsample
            self.df_A = subsampling(self.df_A, subsampling_rate)
            self.df_W = subsampling(self.df_W, subsampling_rate)
            self.df_X = subsampling(self.df_X, subsampling_rate)
            self.df_T = subsampling(self.df_T, subsampling_rate)
            self.df_Y = subsampling(self.df_Y, subsampling_rate)

        if cruise:
            # for each unit and cycle, get only 'cruising' state
            df_all = pd.concat((self.df_A, self.df_W), axis=1)
            df_cruise = df_all.groupby(["unit", "cycle"], group_keys=False).apply(lambda flight: flight[flight["alt"] >= flight['alt'].max() - 1000])
            self.df_X = self.df_X.loc[df_cruise.index]
            self.df_T = self.df_T.loc[df_cruise.index]
            self.df_Y = self.df_Y.loc[df_cruise.index]
            self.df_A = self.df_A.loc[df_cruise.index]
            self.df_W = self.df_W.loc[df_cruise.index]
            del df_all, df_cruise

        if scaling == "standard":
            # standard scaling
            X_mean_1457 = np.array([5.6809174e+02, 1.3297987e+03, 1.6363848e+03, 1.1242614e+03,
            1.2669851e+01, 9.8758097e+00, 1.2862692e+01, 1.5640701e+01,
            2.3289339e+02, 2.3705125e+02, 9.8489847e+00, 1.9629447e+03,
            8.2361680e+03, 2.4945383e+00])
            W_mean_1457 = np.array([1.6362398e+04, 5.4544175e-01,
            6.1369926e+01, 4.8835443e+02])
            X_std_1457 = np.array([2.0833166e+01, 6.7058266e+01, 1.2113522e+02, 6.1629440e+01,
            2.8704438e+00, 2.4181337e+00, 2.9141707e+00, 3.4177959e+00,
            5.7780758e+01, 5.8610283e+01, 2.7480240e+00, 1.8454973e+02,
            2.2242123e+02, 7.6335633e-01])
            W_std_1457 = np.array([8.1254497e+03, 1.2108228e-01,
            1.8272049e+01, 1.9934254e+01])
            self.df_X = (self.df_X - X_mean_1457) / X_std_1457
            self.df_W = (self.df_W - W_mean_1457) / W_std_1457
        elif scaling == "min-max":
            # min-max scaling in [-1, 1] range
            X_min_1457 = np.array([4.9921100e+02, 1.0889111e+03, 1.2294357e+03, 9.1258221e+02,
            6.0976486e+00, 4.4431176e+00, 6.1691909e+00, 7.8722315e+00,
            9.1395287e+01, 9.3320122e+01, 4.5537462e+00, 1.4843353e+03,
            7.4330903e+03, 7.4892521e-01])
            W_min_1457 = np.array([3.0020000e+03, 2.0002499e-01,
            2.3730299e+01, 4.2319705e+02])
            X_max_1457 = np.array([6.3150879e+02, 1.5320127e+03, 1.9813180e+03, 1.3461113e+03,
            2.0096666e+01, 1.4716009e+01, 2.0402708e+01, 2.5905502e+01,
            4.4721826e+02, 4.5370425e+02, 1.6701591e+01, 2.2791799e+03,
            8.8905791e+03, 5.6173835e+00])
            W_max_1457 = np.array([3.5011000e+04, 7.3987198e-01,
            8.7362656e+01, 5.2488281e+02])
            self.df_X = 2 * (self.df_X - X_min_1457) / (X_max_1457 - X_min_1457) - 1
            self.df_W = 2 * (self.df_W - W_min_1457) / (W_max_1457 - W_min_1457) - 1

        self.df_Y /= 100.

        if concepts == "all":
            concepts = [f"{c}-{m}" for c in ("Fan", "LPC", "HPC", "LPT", "HPT") for m in ("E", "F")]
        self.concepts = pd.DataFrame({
            "Fan": self.df_T[['fan_eff_mod', 'fan_flow_mod']].min(axis=1),
            "LPC": self.df_T[['LPC_eff_mod', 'LPC_flow_mod']].min(axis=1),
            "HPC": self.df_T[['HPC_eff_mod', 'HPC_flow_mod']].min(axis=1),
            "LPT": self.df_T[['LPT_eff_mod', 'LPT_flow_mod']].min(axis=1),
            "HPT": self.df_T[['HPT_eff_mod', 'HPT_flow_mod']].min(axis=1),
            "Fan-E": self.df_T['fan_eff_mod'],
            "Fan-F": self.df_T['fan_flow_mod'],
            "LPC-E": self.df_T['LPC_eff_mod'],
            "LPC-F": self.df_T['LPC_flow_mod'],
            "HPC-E": self.df_T['HPC_eff_mod'],
            "HPC-F": self.df_T['HPC_flow_mod'],
            "LPT-E": self.df_T['LPT_eff_mod'],
            "LPT-F": self.df_T['LPT_flow_mod'],
            "HPT-E": self.df_T['HPT_eff_mod'],
            "HPT-F": self.df_T['HPT_flow_mod'],
        })[[c for c in concepts if c not in ["healthy", "Fc"]]]

        # remove non-minimum degradation
        # self.concepts = self.concepts.apply(lambda row: pd.Series([val if val == min(row) else 0 for val in row]), axis=1)

        if binary_concepts:
            self.concepts = self.concepts.apply(binarize_concept)
        else:  # continuous concepts (in [0,1] for BCE loss)
            self.concepts = self.concepts.apply(scale_concept)
        if combined_concepts:  # TODO: treat the case with continuous concepts (use mean instead of all?)
            for combination in combinations(self.concepts.columns.tolist(), 2):
                combination_col = "+".join(combination)
                self.concepts[combination_col] = self.concepts[list(combination)].all(axis=1)
                for c in combination:
                    self.concepts[c] = np.logical_xor(self.concepts[c], self.concepts[combination_col])
        if "healthy" in concepts:
            self.concepts["healthy"] = (self.concepts == 0).all(axis=1)
        if "Fc" in concepts:
            self.concepts = pd.concat((
                self.concepts,
                pd.get_dummies(self.df_A["Fc"]).reindex(columns=[1, 2, 3], fill_value=0)
            ), axis=1)
        # drop constant concepts
        # self.concepts = self.concepts.loc[:, (self.concepts != self.concepts.iloc[0]).any()]
        print("Used concepts:", self.concepts.columns)
        self.concepts = self.concepts.astype(int)

        # available features: measurements (X) + descriptors (W)
        self.X = pd.concat((self.df_X, self.df_W), axis=1)
        del self.df_X, self.df_W

    def __len__(self):
        return self.X.shape[0] // self.stride #Floor total length divided by stride

    def __getitem__(self, i):
        #Sequence is entirely within the data
        if (i * self.stride >= self.window_size - 1):

            unit = self.df_A["unit"].iloc[i * self.stride - self.window_size + 1:i * self.stride + 1]  #Unit vector for desired sequence

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

        return torch.Tensor(x), torch.Tensor(self.df_Y.iloc[i * self.stride]).squeeze(), torch.Tensor(self.concepts.iloc[i * self.stride].to_numpy())
