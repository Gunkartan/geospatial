import os
import sys
import rasterio
import numpy as np
import pandas as pd
from utils.label_extractor import label_extractor
from extraction.extract_crops import process_sentinel
from extraction.extract_water import compute_neighborhood_features
from utils.features import (
    CROP_INDICES,
    CROP_MONTHS,
    CROP_SPECTRAL_FEATURES,
    CROP_TEXTURE_FEATURES,
    PROBABILITY_COLUMNS,
    TEXTURE_WINDOW
)
from scipy.ndimage import binary_erosion

def upstream_probability_grids(
    upstream_probability_file: str,
    shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    upstream_probabilities = pd.read_csv(upstream_probability_file)
    rows = upstream_probabilities['row'].to_numpy()
    cols = upstream_probabilities['col'].to_numpy()
    grids = {}

    for name in [
        'water_probability',
        'building_probability'
    ]:
        grid = np.full(
            shape,
            np.nan,
            dtype='float32'
        )
        grid[rows, cols] = upstream_probabilities[name].to_numpy()
        mean, _ = compute_neighborhood_features(
            grid,
            5
        )
        grids[name] = grid
        grids[f'{name}_mean_5'] = mean

    return grids

def extract_crops(
    year: int,
    remove_water_buildings: bool,
    sampling: bool
) -> None:
    output_name = f'raw_crops_final_pipeline_{year}' if remove_water_buildings else f'raw_crops_standalone_{year}'
    label_file = f'../rasterized/{year}.tif'
    oct_file = f'../raw/47PQQ_{year}-10-31.tif'
    upstream_probability_file = f'../datasets/upstream_probabilities_{year}.csv'

    with rasterio.open(label_file) as label:
        with rasterio.open(oct_file) as tile:
            tile_id = os.path.basename(oct_file).split('_')[0]
            aligned_overlap = label_extractor(
                label,
                tile,
                tile_id
            )

    if remove_water_buildings:
        probability_grids = upstream_probability_grids(
            upstream_probability_file,
            aligned_overlap.shape
        )

    class_map = {
        2101: 'rice',
        2204: 'cassava',
        2205: 'pineapple',
        2302: 'rubber',
        2303: 'oil_palm',
        2403: 'durian',
        2404: 'rambutan',
        2405: 'coconut',
        2407: 'mango',
        2413: 'longan',
        2416: 'jackfruit',
        2419: 'mangosteen',
        2420: 'longkong'
    }
    sampled_rows = []
    sampled_cols = []
    sampled_classes = []
    other_class_ids = np.unique(aligned_overlap)
    other_class_ids = [int(class_id) for class_id in other_class_ids if class_id != 0 and class_id not in class_map]
    rng = np.random.default_rng(42)

    for class_id in [
        *class_map.keys(),
        *other_class_ids
    ]:
        sample_size = 200000 if class_id in class_map else 2000
        mask = aligned_overlap == class_id
        mask = binary_erosion(
            mask,
            iterations=3
        )
        rows, cols = np.where(mask)

        if sampling and len(rows) > sample_size:
            selected = rng.choice(
                len(rows),
                size=sample_size,
                replace=False
            )
            rows = rows[selected]
            cols = cols[selected]

        sampled_rows.append(rows)
        sampled_cols.append(cols)
        sampled_classes.append(np.full(
            len(rows),
            class_id if class_id in class_map else 9999
        ))

    rows = np.concatenate(sampled_rows)
    cols = np.concatenate(sampled_cols)
    data = {
        'row': rows,
        'col': cols,
        'class': np.concatenate(sampled_classes)
    }

    # One month at a time, so only that month's full-size index arrays are in memory.
    for month, date in CROP_MONTHS.items():
        indices = process_sentinel(f'../raw/47PQQ_{year}-{date}.tif')

        for name, index in zip(
            CROP_INDICES,
            indices
        ):
            mean, variance = compute_neighborhood_features(
                index,
                TEXTURE_WINDOW
            )
            data[f'{name}_{month}'] = index[rows, cols]
            data[f'{name}_{month}_mean_{TEXTURE_WINDOW}'] = mean[rows, cols]
            data[f'{name}_{month}_var_{TEXTURE_WINDOW}'] = variance[rows, cols]

        del indices

    if remove_water_buildings:
        for name, grid in probability_grids.items():
            data[name] = grid[rows, cols]

    columns = [
        'row',
        'col',
        'class',
        *CROP_SPECTRAL_FEATURES,
        *CROP_TEXTURE_FEATURES,
        *(PROBABILITY_COLUMNS if remove_water_buildings else [])
    ]
    df = pd.DataFrame(
        data,
        columns=columns
    )
    df.to_csv(
        f'../datasets/{output_name}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError('Please provide values for all the required arguments')

    year = int(sys.argv[1])
    remove_water_buildings = sys.argv[2].lower() == 'true'
    sampling = sys.argv[3].lower() == 'true'
    extract_crops(
        year=year,
        remove_water_buildings=remove_water_buildings,
        sampling=sampling
    )