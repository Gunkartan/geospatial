import os.path
import rasterio
import numpy as np
import pandas as pd
from utils.label_extractor import label_extractor
from scipy.ndimage import binary_erosion

def sample_pixels(
    mask: np.ndarray,
    features: list[np.ndarray],
    sample_size: int,
    buffer_pixels: int = 3
) -> list[list[float]]:
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

if __name__ == '__main__':
    label_file = '../rasterized/2018.tif'
    sentinel_file = '../raw/47PQQ_2018-10-31.tif'
    label = rasterio.open(label_file)
    tile = rasterio.open(sentinel_file)
    tile_id = os.path.basename(sentinel_file).split('_')[0]
    aligned_overlap = label_extractor(
        label,
        tile,
        tile_id
    )
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
    features = [
        ndvi,
        evi,
        ndwi,
        mtci,
        swir_long
    ]
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
        2420: 'longkong'
    }
    others = 9999

    for class_id in class_map:
        mask = aligned_overlap == class_id
        rows = sample_pixels(
            mask,
            features,
            samples_per_class
        )

        for row in rows:
            row.append(class_id)
            dataset.append(row)

    known_mask = np.isin(
        aligned_overlap,
        list(class_map.keys())
    )
    water_mask = np.char.startswith(
        aligned_overlap.astype(str),
        '4'
    )
    building_mask = np.char.startswith(
        aligned_overlap.astype(str),
        '1'
    )
    others_mask = ~(known_mask | water_mask | building_mask)
    rows = sample_pixels(
        others_mask,
        features,
        samples_per_class
    )

    for row in rows:
        row.append(others)
        dataset.append(row)

    columns = [
        'ndvi',
        'evi',
        'ndwi',
        'mtci',
        'swir_long',
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