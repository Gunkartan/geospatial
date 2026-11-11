import os
import sys
import numpy as np
import pandas as pd
import rasterio
from utils.label_extractor import label_extractor
from utils.features import CROP_MONTHS, UPSTREAM_FEATURES, UPSTREAM_MONTHLY_FEATURES
from extraction.extract_water import compute_neighborhood_features, process_water_features

CHUNK_SIZE = 2_000_000

def split_feature(feature: str) -> tuple[
    str,
    str
]:
    for suffix in (
        '_variance',
        '_mean'
    ):
        if feature.endswith(suffix):
            return feature[:-len(suffix)], suffix

    return feature, ''

def extract_water(year: int) -> None:
    """Upstream features (UPSTREAM_FEATURES for Oct, Nov and Dec) for every labelled pixel.

    The 3x3 neighbourhood mean and variance are computed over the whole tile, so
    there are no edge effects at block borders.
    """
    label_file = f'../rasterized/{year}.tif'
    oct_file = f'../raw/47PQQ_{year}-10-31.tif'

    with rasterio.open(label_file) as label:
        with rasterio.open(oct_file) as tile:
            tile_id = os.path.basename(oct_file).split('_')[0]
            aligned_overlap = label_extractor(
                label,
                tile,
                tile_id
            )

    rows, cols = np.where(aligned_overlap != 0)
    classes = aligned_overlap[rows, cols]
    del aligned_overlap
    needed = {}

    for feature in UPSTREAM_FEATURES:
        base, suffix = split_feature(feature)
        needed.setdefault(base, set()).add(suffix)

    column = {name: i for i, name in enumerate(UPSTREAM_MONTHLY_FEATURES)}
    features = np.empty(
        (
            len(rows),
            len(UPSTREAM_MONTHLY_FEATURES)
        ),
        dtype='float32'
    )

    for month, date in CROP_MONTHS.items():
        indices = process_water_features(f'../raw/{tile_id}_{year}-{date}.tif')

        for base, suffixes in needed.items():
            arr = indices.pop(base)
            values = {'': arr[rows, cols]}

            if suffixes - {''}:
                mean, variance = compute_neighborhood_features(
                    arr,
                    3
                )
                values['_mean'] = mean[rows, cols]
                values['_variance'] = variance[rows, cols]
                del mean, variance

            for suffix in suffixes:
                value = values[suffix]
                value[~np.isfinite(value)] = np.nan
                features[:, column[f'{base}{suffix}_{month}']] = value

            del arr, values

        del indices
        print(f'{year} {month} features done', flush=True)

    output_file = f'../datasets/raw_water_final_pipeline_{year}.csv'

    for start in range(
        0,
        len(rows),
        CHUNK_SIZE
    ):
        end = start + CHUNK_SIZE
        df = pd.DataFrame(
            features[start:end],
            columns=UPSTREAM_MONTHLY_FEATURES
        ).round(3)
        df.insert(0, 'class', classes[start:end])
        df.insert(0, 'col', cols[start:end])
        df.insert(0, 'row', rows[start:end])
        df.to_csv(
            output_file,
            mode='w' if start == 0 else 'a',
            header=start == 0,
            index=False
        )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    extract_water(year)
