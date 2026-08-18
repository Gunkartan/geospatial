import os.path
import rasterio
import numpy as np
import pandas as pd
from utils.label_extractor import label_extractor
from scipy.ndimage import binary_erosion

def sample_pixels(
    mask: np.ndarray,
    sample_size: int,
    buffer_pixels: int = 3
) -> np.ndarray:
    mask = binary_erosion(
        mask,
        iterations=buffer_pixels
    )
    idx = np.column_stack(np.where(mask))

    if len(idx) > sample_size:
        idx = idx[np.random.choice(
            len(idx),
            sample_size,
            replace=False
        )]

    return idx

def extract_features(
    idx: np.ndarray,
    features: list[np.ndarray]
) -> list[list[float]]:
    rows = []

    for r, c in idx:
        row = []

        for feature in features:
            row.append(feature[
                r,
                c
            ])

        rows.append(row)

    return rows

def process_sentinel(sentinel_file: str) -> list[np.ndarray]:
    tile = rasterio.open(sentinel_file)
    blue = tile.read(1).astype('float32')
    green = tile.read(2).astype('float32')
    red = tile.read(3).astype('float32')
    re_early = tile.read(4).astype('float32')
    re_mid = tile.read(5).astype('float32')
    nir = tile.read(7).astype('float32')
    swir = tile.read(8).astype('float32')
    swir_long = tile.read(9).astype('float32')
    ndvi = (nir - red) / (nir + red)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    ndwi = (green - swir) / (green + swir)
    mtci = (re_mid - re_early) / (re_early - red)

    return [
        ndvi,
        evi,
        ndwi,
        mtci,
        swir_long
    ]

if __name__ == '__main__':
    label_file = '../rasterized/2018.tif'
    oct_file = '../raw/47PQQ_2018-10-31.tif'
    nov_file = '../raw/47PQQ_2018-11-30.tif'
    dec_file = '../raw/47PQQ_2018-12-31.tif'
    label = rasterio.open(label_file)
    oct = rasterio.open(oct_file)
    tile_id = os.path.basename(oct_file).split('_')[0]
    aligned_overlap = label_extractor(
        label,
        oct,
        tile_id
    )
    oct_features = process_sentinel(oct_file)
    nov_features = process_sentinel(nov_file)
    dec_features = process_sentinel(dec_file)
    samples_per_class = 200000
    dataset = []
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

    for class_id in class_map:
        mask = ~known_mask if class_id == 9999 else aligned_overlap == class_id
        idx = sample_pixels(
            mask,
            samples_per_class
        )
        oct_rows = extract_features(
            idx,
            oct_features
        )
        nov_rows = extract_features(
            idx,
            nov_features
        )
        dec_rows = extract_features(
            idx,
            dec_features
        )

        for oct_row, nov_row, dec_row in zip(
            oct_rows,
            nov_rows,
            dec_rows
        ):
            row = oct_row + nov_row + dec_row
            row.append(class_id)
            dataset.append(row)

    columns = [
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
        'swir_long_dec',
        'class'
    ]
    df = pd.DataFrame(
        dataset,
        columns=columns
    )
    df.to_csv(
        '../datasets/raw_crops.csv',
        index=False
    )