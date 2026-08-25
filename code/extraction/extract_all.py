import os
import sys
import numpy as np
import pandas as pd
import rasterio
from utils.label_extractor import label_extractor
from extraction.extract_water import compute_neighborhood_features, process_water_features
from extraction.extract_crops import process_sentinel

def update_reservoir(
    reservoir: list[np.ndarray],
    candidates: np.ndarray,
    sample_size: int,
    seen_count: int,
    rng: np.random.Generator
) -> int:
    for candidate in candidates:
        seen_count += 1

        if len(reservoir) < sample_size:
            reservoir.append(candidate)

        else:
            index = rng.integers(
                0,
                seen_count
            )

            if index < sample_size:
                reservoir[index] = candidate

    return seen_count

def get_sampling_classes(
    labels: np.ndarray,
    crop_classes: set[int]
) -> np.ndarray:
    classes = np.zeros(
        labels.shape,
        dtype=np.int32
    )
    valid = labels != 0
    water = np.char.startswith(
        labels.astype(str),
        '4'
    )
    building = np.char.startswith(
        labels.astype(str),
        '1'
    )
    crop = np.isin(
        labels,
        list(crop_classes)
    )
    known = valid & (water | building | crop)
    classes[known] = labels[known]
    classes[valid & ~known] = 9999

    return classes

def write_csv(
    data: dict[
        str,
        np.ndarray
    ],
    columns: list[str],
    year: int,
    first_write: bool
) -> None:
    df = pd.DataFrame(data)
    df = df[columns]
    df = df.round(3)

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

def group_samples_by_block(
    reservoirs: dict[
        int,
        list[np.ndarray]
    ],
    block_size: int
) -> dict[
    tuple[
        int,
        int
    ],
    list[tuple[
        np.ndarray,
        int
    ]]
]:
    samples_by_block = {}

    for class_id, indices in reservoirs.items():
        indices = np.asarray(
            indices,
            dtype=np.int32
        )

        if len(indices) == 0:
            continue

        block_rows = (indices[
            :,
            0
        ] // block_size) * block_size
        block_cols = (indices[
            :,
            1
        ] // block_size) * block_size
        block_keys = np.column_stack([
            block_rows,
            block_cols
        ])
        unique_blocks = np.unique(
            block_keys,
            axis=0
        )

        for block_row, block_col in unique_blocks:
            block_row = int(block_row)
            block_col = int(block_col)
            mask = ((block_rows == block_row) & (block_cols == block_col))
            samples = indices[mask]
            key = (
                block_row,
                block_col
            )

            if key not in samples_by_block:
                samples_by_block[key] = []

            samples_by_block[key].append((
                samples,
                class_id
            ))

    return samples_by_block

def extract_all(year: int) -> None:
    label_file = f'../rasterized/{year}.tif'
    crop_classes = {
        2101,
        2204,
        2205,
        2302,
        2303,
        2403,
        2404,
        2405,
        2407,
        2413,
        2416,
        2419,
        2420
    }
    oct_file = f'../raw/47PQQ_{year}-10-31.tif'
    nov_file = f'../raw/47PQQ_{year}-11-30.tif'
    dec_file = f'../raw/47PQQ_{year}-12-31.tif'
    sample_size = 200000
    block_size = 1024
    rng = np.random.default_rng(42)
    reservoirs = {}
    seen_counts = {}

    with rasterio.open(label_file) as label:
        with rasterio.open(oct_file) as tile:
            tile_id = os.path.basename(oct_file).split('_')[0]
            aligned_overlap = label_extractor(
                label,
                tile,
                tile_id
            )

    water_features = process_water_features(oct_file)
    oct_crop_features = process_sentinel(oct_file)
    nov_crop_features = process_sentinel(nov_file)
    dec_crop_features = process_sentinel(dec_file)
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
            label_block = aligned_overlap[
                row_start:row_end,
                col_start:col_end
            ]
            sampling_classes = get_sampling_classes(
                label_block,
                crop_classes
            )
            valid_block = np.ones(
                label_block.shape,
                dtype=bool
            )
            water_statistics = {}

            for name, arr in water_features.items():
                block = arr[
                    row_start:row_end,
                    col_start:col_end
                ]
                mean, var = compute_neighborhood_features(
                    block,
                    3
                )
                water_statistics[name] = (
                    block,
                    mean,
                    var
                )
                valid_block &= np.isfinite(block)
                valid_block &= np.isfinite(mean)
                valid_block &= np.isfinite(var)

            for _, features in crop_feature_sets:
                for arr in features:
                    block = arr[
                        row_start:row_end,
                        col_start:col_end
                    ]
                    valid_block &= np.isfinite(block)

            sampling_classes[~valid_block] = 0

            for class_id in np.unique(sampling_classes):
                if class_id == 0:
                    continue

                if class_id not in reservoirs:
                    reservoirs[class_id] = []
                    seen_counts[class_id] = 0

                local_idx = np.column_stack(np.where(sampling_classes == class_id))
                global_idx = local_idx + np.array([
                    row_start,
                    col_start
                ])
                seen_counts[class_id] = update_reservoir(
                    reservoirs[class_id],
                    global_idx,
                    sample_size,
                    seen_counts[class_id],
                    rng
                )

    samples_by_block = group_samples_by_block(
        reservoirs,
        block_size
    )

    if not samples_by_block:
        raise ValueError('No valid samples were found')

    first_write = True

    for (
        row_start,
        col_start
    ), sample_groups in samples_by_block.items():
        row_end = min(
            row_start + block_size,
            num_rows
        )
        col_end = min(
            col_start + block_size,
            num_cols
        )
        water_statistics = {}

        for name, arr in water_features.items():
            block = arr[
                row_start:row_end,
                col_start:col_end
            ]
            mean, var = compute_neighborhood_features(
                block,
                3
            )
            water_statistics[name] = (
                block,
                mean,
                var
            )

        sampled_indices = np.concatenate([indices for indices, _ in sample_groups])
        sampled_classes = np.concatenate([np.full(
            len(indices),
            class_id,
            dtype=np.int32
        ) for indices, class_id in sample_groups])
        rows = sampled_indices[
            :,
            0
        ] - row_start
        cols = sampled_indices[
            :,
            1
        ] - col_start
        data = {}

        for name, (
            block,
            mean,
            var
        ) in water_statistics.items():
            data[name] = block[
                rows,
                cols
            ]
            data[f'{name}_mean'] = mean[
                rows,
                cols
            ]
            data[f'{name}_variance'] = var[
                rows,
                cols
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
                data[f'{name}_{month}'] = block[
                    rows,
                    cols
                ]

        data['class'] = sampled_classes
        write_csv(
            data,
            columns + ['class'],
            year,
            first_write
        )
        first_write = False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')
    
    year = int(sys.argv[1])
    extract_all(year)