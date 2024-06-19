import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List


def generate_data_with_label(
        n_samples: int,
        mean_vector: List[float],
        cov_matrix: List[List[float]],
        label: int
):
    x = np.random.multivariate_normal(mean_vector, cov_matrix, n_samples)
    df = pd.DataFrame(x, columns=[f"x_{i}" for i in range(1, len(mean_vector) + 1)])
    df["label"] = label
    return df


def set_up_df(
        n_samples: int,
        normal_mean_vector: List[float],
        normal_cov_matrix: List[List[float]],
        abnormal_mean_vector: List[float],
        abnormal_cov_matrix: List[List[float]],
):
    df_normal = generate_data_with_label(
        n_samples, normal_mean_vector, normal_cov_matrix, 0
    )
    df_abnormal = generate_data_with_label(
        n_samples, abnormal_mean_vector, abnormal_cov_matrix, 1
    )

    df = pd.concat((df_normal, df_abnormal), ignore_index=True)
    df.to_csv("data.csv", index=False)
    return df


class Data(Dataset):
    def __init__(
            self,
            n_samples: int,
            normal_mean_vector: List[float],
            normal_cov_matrix: List[List[float]],
            abnormal_mean_vector: List[float],
            abnormal_cov_matrix: List[List[float]],
    ):
        df = set_up_df(
            n_samples,
            normal_mean_vector,
            normal_cov_matrix,
            abnormal_mean_vector, abnormal_cov_matrix
        )

        self.x = df.drop(columns=["label"]).values
        self.y = df["label"].values

    def __getitem__(self, item):
        return self.x[item, :], self.y[item]

    def __len__(self):
        return self.y.__len__()


if __name__ == "__main__":
    n = 1000
    m1 = [0, 0]
    cov1 = [[1, -1], [-1, 1]]
    m2 = [0, 0]
    cov2 = [[10, -1], [-1, 10]]
    df = Data(n, m1, cov1, m2, cov2)
