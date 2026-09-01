import os
import sys
import numpy as np
import pandas as pd
import rasterio
from utils.label_extractor import label_extractor
from extraction.extract_water import compute_neighborhood_features, process_water_features

def create_csv(
    features: dict[
        str,
        np.ndarray
    ],
    columns: list[str],
    year: int,
    first_write: bool
) -> None:
    df = pd.DataFrame(
        features,
        columns=columns
    )
    df = df.round(3)

    if first_write:
        df.to_csv(
            f'../datasets/raw_buildings_final_pipeline_{year}.csv',
            index=False
        )

    else:
        df.to_csv(
            f'../datasets/raw_buildings_final_pipeline_{year}.csv',
            mode='a',
            header=False,
            index=False
        )

def extract_buildings(year: int) -> None:
    label_file = f'../rasterized/{year}.tif'
    sentinel_file = f'../raw/47PQQ_{year}-10-31.tif'
    water_indices_file = f'../datasets/water_indices_{year}.csv'
    water_indices = pd.read_csv(water_indices_file)
    water_set = set(zip(
        water_indices['row'],
        water_indices['col']
    ))

    with rasterio.open(label_file) as label:
        with rasterio.open(sentinel_file) as tile:
            tile_id = os.path.basename(sentinel_file).split('_')[0]
            aligned_overlap = label_extractor(
                label,
                tile,
                tile_id
            )

    water_features = process_water_features(sentinel_file)
    block_size = 1024
    num_rows, num_cols = next(iter(water_features.values())).shape
    water_feature_names = []

    for name in water_features.keys():
        water_feature_names.extend([
            name,
            f'{name}_mean',
            f'{name}_variance'
        ])

    columns = [
        'row',
        'col',
        'class'
    ] + water_feature_names
    first_write = True

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
            label_block = aligned_overlap[
                row_start:row_end,
                col_start:col_end
            ]
            mask = label_block != 0
            local_rows, local_cols = np.where(mask)

            if len(local_rows) == 0:
                continue

            rows = local_rows + row_start
            cols = local_cols + col_start
            keep = np.array([(
                row,
                col
            ) not in water_set for row, col in zip(
                rows,
                cols
            )])
            local_rows = local_rows[keep]
            local_cols = local_cols[keep]
            rows = rows[keep]
            cols = cols[keep]

            if len(rows) == 0:
                continue

            data = {
                'row': rows,
                'col': cols,
                'class': label_block[
                    local_rows,
                    local_cols
                ]
            }

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

            create_csv(
                data,
                columns,
                year,
                first_write
            )
            first_write = False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    extract_buildings(year)