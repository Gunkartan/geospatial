import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../datasets/raw_water.csv')
    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    df = df.dropna().reset_index(drop=True)
    df.to_csv(
        '../datasets/preprocessed_water.csv',
        index=False
    )