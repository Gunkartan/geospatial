import sys
import numpy as np
import pandas as pd

def preprocess_crops(year: int) -> None:
    df = pd.read_csv(f'../datasets/raw_crops_final_pipeline_{year}.csv')
    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    df = df.dropna().reset_index(drop=True)
    df.to_csv(
        f'../datasets/preprocessed_crops_final_pipeline_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    preprocess_crops(year)