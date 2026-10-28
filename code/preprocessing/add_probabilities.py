import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from extraction.extract_water import compute_neighborhood_features, process_water_features

def add_probabilities(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_crops_standalone_{year}.csv')
    sentinel_file = f'../raw/47PQQ_{year}-10-31.tif'
    water_model = xgb.XGBClassifier()
    water_model.load_model('../models/water.json')
    building_model = xgb.XGBClassifier()
    building_model.load_model('../models/buildings.json')
    water_features = process_water_features(sentinel_file)
    water_feature_names = []

    for name in water_features.keys():
        water_feature_names.extend([
            name,
            f'{name}_mean',
            f'{name}_variance'
        ])

    # The neighborhood features are computed per block, the same way as in the water and building extraction
    block_size = 1024
    num_rows, num_cols = next(iter(water_features.values())).shape
    water_probability = np.zeros(
        len(df),
        dtype='float32'
    )
    building_probability = np.zeros(
        len(df),
        dtype='float32'
    )
    block_ids = pd.DataFrame({
        'block_row': df['row'] // block_size,
        'block_col': df['col'] // block_size
    })

    for (block_row, block_col), group in block_ids.groupby([
        'block_row',
        'block_col'
    ]):
        row_start = block_row * block_size
        col_start = block_col * block_size
        row_end = min(
            row_start + block_size,
            num_rows
        )
        col_end = min(
            col_start + block_size,
            num_cols
        )
        positions = group.index.to_numpy()
        local_rows = df['row'].to_numpy()[positions] - row_start
        local_cols = df['col'].to_numpy()[positions] - col_start
        data = {}

        for name, arr in water_features.items():
            block = arr[
                row_start:row_end,
                col_start:col_end
            ]
            mean, variance = compute_neighborhood_features(
                block,
                3
            )
            data[name] = block[
                local_rows,
                local_cols
            ]
            data[f'{name}_mean'] = mean[
                local_rows,
                local_cols
            ]
            data[f'{name}_variance'] = variance[
                local_rows,
                local_cols
            ]

        features = pd.DataFrame(
            data,
            columns=water_feature_names
        )
        features = features.round(3)
        # XGBoost treats NaN as missing, so pixels with invalid water features still get probabilities
        features = features.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        water_probability[positions] = water_model.predict_proba(features)[
            :,
            1
        ]
        building_probability[positions] = building_model.predict_proba(features)[
            :,
            1
        ]

    df['water_probability'] = water_probability
    df['building_probability'] = building_probability
    df.to_csv(
        f'../datasets/preprocessed_crops_probabilities_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    add_probabilities(year)
