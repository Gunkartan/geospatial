import os
import sys
import rasterio
import numpy as np
import pandas as pd
from utils.label_extractor import label_extractor
from extraction.extract_crops import process_sentinel

def extract_crops(year: int) -> None:
    label_file = f'../rasterized/{year}.tif'
    oct_file = f'../raw/47PQQ_{year}-10-31.tif'
    nov_file = f'../raw/47PQQ_{year}-11-30.tif'
    dec_file = f'../raw/47PQQ_{year}-12-31.tif'
    water_indices_file = f'../datasets/water_indices_{year}.csv'
    building_indices_file = f'../datasets/building_indices_{year}.csv'
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
        2420: 'longkong',
        9999: 'others'
    }
    known_mask = np.isin(
        aligned_overlap,
        list(class_map.keys())
    )
    rows, cols = np.where(known_mask)
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
    rows = rows[keep]
    cols = cols[keep]
    dataset = []

    for row, col in zip(
        rows,
        cols
    ):
        label = aligned_overlap[
            row,
            col
        ]

        if label in class_map:
            class_id = label

        else:
            class_id = 9999

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
    df = df.round(3)
    df.to_csv(
        f'../datasets/raw_crops_final_pipeline_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    extract_crops(year)