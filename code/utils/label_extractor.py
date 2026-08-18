import os
import numpy
import rasterio
from rasterio.windows import from_bounds

def label_extractor(
    label_raster: rasterio.DatasetReader,
    tile_raster: rasterio.DatasetReader,
    tile_id: str
) -> numpy.ndarray | None:
    label_bounds = label_raster.bounds
    tile_bounds = tile_raster.bounds
    intersection = (
        max(
            label_bounds.left,
            tile_bounds.left
        ),
        max(
            label_bounds.bottom,
            tile_bounds.bottom
        ),
        min(
            label_bounds.right,
            tile_bounds.right
        ),
        min(
            label_bounds.top,
            tile_bounds.top
        )
    )

    if intersection[0] < intersection[2] and intersection[1] < intersection[3]:
        label_window = from_bounds(
            *intersection,
            transform=label_raster.transform
        )
        overlap_data = label_raster.read(
            1,
            window=label_window
        )
        print(overlap_data.shape)
        raster_window = from_bounds(
            *intersection,
            transform=tile_raster.transform
        )
        aligned_overlap = numpy.full(
            (
                tile_raster.height,
                tile_raster.width
            ),
            0,
            dtype=label_raster.dtypes[0]
        )
        row_start = int(raster_window.row_off)
        row_end = row_start + overlap_data.shape[0]
        col_start = int(raster_window.col_off)
        col_end = col_start + overlap_data.shape[1]
        aligned_overlap[
            row_start:row_end,
            col_start:col_end
        ] = overlap_data
        profile = tile_raster.profile
        profile.update({
            'driver': 'GTiff',
            'count': 1,
            'nodata': 0,
            'compress': 'LZW'
        })
        os.makedirs(
            '../../label_files/',
            exist_ok=True
        )

        with rasterio.open(
            f'../../label_files/label_{tile_id}.tif',
            'w',
            **profile
        ) as dest:
            dest.write(
                aligned_overlap,
                1
            )

        return aligned_overlap

    return None