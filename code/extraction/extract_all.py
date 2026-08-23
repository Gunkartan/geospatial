import sys
import numpy as np
import pandas as pd
from extraction.extract_water import compute_neighborhood_features, process_water_features
from extraction.extract_crops import process_sentinel

def create_csv(
    features: dict[
        str,
        np.ndarray
    ],
    columns: list[str],
    year: int,
    first_write: bool
) -> None:
    df = pd.DataFrame(features)
    df = df[columns]

    if first_write:
        df.to_csv(
            f'../datasets/raw_all_{year}.csv',
            index=False
        )

    else:
        df.to_csv(
            f'../datasets/raw_all_{year}.csv',
            mode='a',
            header=False,
            index=False
        )

def extract_all(year: int) -> None:
    oct_file = f'../raw/47PQQ_{year}-10-31.tif'
    nov_file = f'../raw/47PQQ_{year}-11-30.tif'
    dec_file = f'../raw/47PQQ_{year}-12-31.tif'
    water_features = process_water_features(oct_file)
    oct_crop_features = process_sentinel(oct_file)
    nov_crop_features = process_sentinel(nov_file)
    dec_crop_features = process_sentinel(dec_file)
    block_size = 1024
    num_rows, num_cols = next(iter(water_features.values())).shape
    water_feature_names = []

    for name in water_features.keys():
        water_feature_names.extend([
            name,
            f'{name}_mean',
            f'{name}_variance'
        ])

    crop_feature_names = [
        'ndvi_oct',
        'evi_oct',
        'ndwi_oct',
        'mtci_oct',
        'swir_long_oct',
        'ndvi_nov',
        'evi_nov',
        'ndwi_nov',
        'mtci_nov',
        'swir_long_nov',
        'ndvi_dec',
        'evi_dec',
        'ndwi_dec',
        'mtci_dec',
        'swir_long_dec'
    ]
    columns = water_feature_names + crop_feature_names

    for row_start in range(
        0,
        num_rows,
        block_size
    ):
        for col_start in range(
            0,
            num_cols,
            block_size
        ):
            row_end = min(
                row_start + block_size,
                num_rows
            )
            col_end = min(
                col_start + block_size,
                num_cols
            )
            data = {}

            for name, arr in water_features.items():
                block = arr[
                    row_start:row_end,
                    col_start:col_end
                ]
                mean, var = compute_neighborhood_features(
                    block,
                    3
                )
                data[name] = block.ravel()
                data[f'{name}_mean'] = mean.ravel()
                data[f'{name}_variance'] = var.ravel()

            crop_feature_sets = [
                (
                    'oct',
                    oct_crop_features
                ),
                (
                    'nov',
                    nov_crop_features
                ),
                (
                    'dec',
                    dec_crop_features
                )
            ]
            crop_indices = [
                'ndvi',
                'evi',
                'ndwi',
                'mtci',
                'swir_long'
            ]

            for month, features in crop_feature_sets:
                for name, arr in zip(
                    crop_indices,
                    features
                ):
                    block = arr[
                        row_start:row_end,
                        col_start:col_end
                    ]
                    data[f'{name}_{month}'] = block.ravel()

            create_csv(
                data,
                columns,
                year,
                row_start == 0 and col_start == 0
            )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')
    
    year = int(sys.argv[1])
    extract_all(year)