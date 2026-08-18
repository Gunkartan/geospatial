import os.path
import rasterio
import numpy as np
import pandas as pd
from utils.label_extractor import label_extractor
from scipy.ndimage import uniform_filter

def compute_neighborhood_features(
    arr: np.ndarray,
    size: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = uniform_filter(
        arr,
        size=size,
        mode='reflect'
    )
    mean_square = uniform_filter(
        arr ** 2,
        size=size,
        mode='reflect'
    )
    var = mean_square - mean ** 2

    return mean, var

def extract_raw_vals(
    aligned_overlap: np.ndarray,
    index_dict: dict[
        str,
        np.ndarray
    ]
) -> np.ndarray:
    mask = aligned_overlap != 0
    raw_vals = aligned_overlap[mask].astype(int)
    labels = np.array([1 if str(val).startswith('4') else 0 for val in raw_vals])
    features = [raw_vals]

    for _, arr in index_dict.items():
        features.append(arr[mask])
        mean, var = compute_neighborhood_features(
            arr,
            3
        )
        features.append(mean[mask])
        features.append(var[mask])

    features = np.column_stack(features + [labels])

    return features

def create_csv(
    features: np.ndarray,
    cols: list[str],
    first_write: bool
) -> None:
    df = pd.DataFrame(
        features,
        columns=cols
    )
    df = df.round(3)

    if first_write:
        df.to_csv(
            'raw.csv',
            index=False
        )

    else:
        df.to_csv(
            'raw.csv',
            mode='a',
            header=False,
            index=False
        )

if __name__ == '__main__':
    label_file = '../../rasterized/2018.tif'
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
    nir = tile.read(7).astype('float32')
    swir = tile.read(8).astype('float32')
    swir_long = tile.read(9).astype('float32')
    ndvi = (nir - red) / (nir + red)
    ndwi = (green - nir) / (green + nir)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    ndbi = (swir - nir) / (swir + nir)
    mndwi = (green - swir) / (green + swir)
    bsi = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue))
    ndsi = (swir - green) / (swir + green)
    ndti = (swir - swir_long) / (swir + swir_long)
    si = (swir - nir) / (swir + nir)
    ibi = ((ndbi - ((ndvi + mndwi) / 2)) / (ndbi + ((ndvi + mndwi) / 2)))
    ndbsi = (si + ibi) / 2
    bi = (red + green + blue) / 3
    rbc = (red - blue) / (red + blue)
    grri = green / red
    rsr = red / swir_long
    rgri = red / green
    snr = swir / nir
    bri = (red + green) / (nir + swir)
    index_dict = {
        'ndvi': ndvi,
        'ndwi': ndwi,
        'evi': evi,
        'ndbi': ndbi,
        'mndwi': mndwi,
        'bsi': bsi,
        'ndsi': ndsi,
        'ndti': ndti,
        'ndbsi': ndbsi,
        'bi': bi,
        'rbc': rbc,
        'grri': grri,
        'rsr': rsr,
        'rgri': rgri,
        'snr': snr,
        'bri': bri
    }
    block_size = 1024
    num_rows, num_cols = aligned_overlap.shape
    columns = ['labels']

    for name in index_dict.keys():
        columns.extend([name, f'{name} mean', f'{name} variance'])

    columns.append('water')

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
            aligned_block = aligned_overlap[
                row_start:row_end,
                col_start:col_end
            ]
            index_block = {}

            for name, arr in index_dict.items():
                index_block[name] = arr[
                    row_start:row_end,
                    col_start:col_end
                ]

            feature_block = extract_raw_vals(
                aligned_block,
                index_block
            )
            create_csv(
                feature_block,
                columns,
                row_start == 0 and col_start == 0
            )