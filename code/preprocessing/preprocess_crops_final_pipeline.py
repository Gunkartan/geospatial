import sys
import numpy as np
import pandas as pd
from utils.features import CROP_SPECTRAL_FEATURES

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
    # Rows with missing features are kept (October is missing for about half of 2020;
    # the crop models are trained to fall back on Nov/Dec). Only rows with no
    # spectral data in any month are dropped.
    spectral = [column for column in CROP_SPECTRAL_FEATURES if column in df.columns]
    df = df[df[spectral].notna().any(axis=1)].reset_index(drop=True)
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