import sys
import numpy as np
import pandas as pd

def preprocess_crops(
    year: int,
    standalone: bool
) -> None:
    filename = f'raw_crops_standalone_{year}' if standalone else f'raw_crops_final_pipeline_{year}'
    output_name = f'preprocessed_crops_standalone_{year}' if standalone else f'preprocessed_crops_final_pipeline_{year}'
    df = pd.read_csv(f'../datasets/{filename}.csv')
    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    df = df.dropna().reset_index(drop=True)
    df.to_csv(
        f'../datasets/{output_name}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Please provide values for all the required arguments')

    year = int(sys.argv[1])
    standalone = sys.argv[2].lower() == 'true'
    preprocess_crops(
        year=year,
        standalone=standalone
    )