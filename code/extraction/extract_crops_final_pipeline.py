import os
import sys
import rasterio
import numpy as np
import pandas as pd
from utils.label_extractor import label_extractor
from extraction.extract_crops import process_sentinel
from scipy.ndimage import binary_erosion

def extract_crops(
    year: int,
    remove_water_buildings: bool,
    sampling: bool
) -> None:
    output_name = f'raw_crops_final_pipeline_{year}' if remove_water_buildings else f'raw_crops_standalone_{year}'
    label_file = f'../rasterized/{year}.tif'
    oct_file = f'../raw/47PQQ_{year}-10-31.tif'
    nov_file = f'../raw/47PQQ_{year}-11-30.tif'
    dec_file = f'../raw/47PQQ_{year}-12-31.tif'
    water_indices_file = f'../datasets/water_indices_{year}.csv'
    building_indices_file = f'../datasets/building_indices_{year}.csv'
    excluded_file = f'../datasets/raw_crops_excluded_{year}.csv'

    if remove_water_buildings:
        water_indices = pd.read_csv(water_indices_file)
        building_indices = pd.read_csv(building_indices_file)
        water_set = set(zip(
            water_indices['row'],
            water_indices['col']
        ))
        building_set = set(zip(
            building_indices['row'],
            building_indices['col']
        ))

    with rasterio.open(label_file) as label:
        with rasterio.open(oct_file) as tile:
            tile_id = os.path.basename(oct_file).split('_')[0]
            aligned_overlap = label_extractor(
                label,
                tile,
                tile_id
            )

    oct_features = process_sentinel(oct_file)
    nov_features = process_sentinel(nov_file)
    dec_features = process_sentinel(dec_file)
    feature_sets = [
        oct_features,
        nov_features,
        dec_features
    ]
    feature_names = [
        'ndvi',
        'evi',
        'ndwi',
        'mtci',
        'swir_long'
    ]
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
    dataset = []
    excluded_dataset = []

    for class_id in [
        *class_map.keys(),
        9999
    ]:
        if class_id == 9999:
            mask = (aligned_overlap != 0) & ~np.isin(
                aligned_overlap,
                list(class_map.keys())
            )

        else:
            mask = aligned_overlap == class_id

        mask = binary_erosion(
            mask,
            iterations=3
        )
        rows, cols = np.where(mask)

        if remove_water_buildings:
            keep = np.array([(
                row,
                col
            ) not in water_set and (
                row,
                col
            ) not in building_set for row, col in zip(
                rows,
                cols
            )])
            excluded_rows = rows[~keep]
            excluded_cols = cols[~keep]

            for row, col in zip(
                excluded_rows,
                excluded_cols
            ):
                excluded_dataset.append({
                    'row': row,
                    'col': col,
                    'class': class_id
                })

            rows = rows[keep]
            cols = cols[keep]

        for row, col in zip(
            rows,
            cols
        ):
            data = {
                'row': row,
                'col': col
            }

            for month, features in zip(
                [
                    'oct',
                    'nov',
                    'dec'
                ],
                feature_sets
            ):
                for name, feature in zip(
                    feature_names,
                    features
                ):
                    data[f'{name}_{month}'] = feature[
                        row,
                        col
                    ]

            data['class'] = class_id
            dataset.append(data)

    columns = [
        'row',
        'col',
        'class',
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
    df = pd.DataFrame(
        dataset,
        columns=columns
    )

    if sampling:
        samples_per_class = 200000
        sampled_datasets = []

        for class_id, group in df.groupby('class'):
            if class_id != 9999 and len(group) > samples_per_class:
                group = group.sample(
                    n=samples_per_class,
                    random_state=42
                )

            sampled_datasets.append(group)

        df = pd.concat(
            sampled_datasets,
            ignore_index=True
        )

    df.to_csv(
        f'../datasets/{output_name}.csv',
        index=False
    )

    if remove_water_buildings:
        excluded_df = pd.DataFrame(
            excluded_dataset,
            columns=[
                'row',
                'col',
                'class'
            ]
        )
        excluded_df.to_csv(
            excluded_file,
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