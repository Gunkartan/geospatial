import sys
import numpy as np
import pandas as pd

def preprocess_water(year: int) -> None:
    # Streamed in chunks: the 3-month upstream table does not fit in memory as float64.
    chunks = pd.read_csv(
        f'../datasets/raw_water_final_pipeline_{year}.csv',
        chunksize=2_000_000
    )

    for i, df in enumerate(chunks):
        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        df.to_csv(
            f'../datasets/preprocessed_water_final_pipeline_{year}.csv',
            mode='w' if i == 0 else 'a',
            header=i == 0,
            index=False
        )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    preprocess_water(year)
