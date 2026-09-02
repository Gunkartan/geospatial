import sys
from extraction.extract_water_final_pipeline import extract_water
from extraction.extract_buildings_final_pipeline import extract_buildings
from extraction.extract_crops_final_pipeline import extract_crops
from preprocessing.preprocess_water_final_pipeline import preprocess_water
from preprocessing.preprocess_buildings_final_pipeline import preprocess_buildings
from preprocessing.preprocess_crops_final_pipeline import preprocess_crops
from inference.infer_water import infer_water
from inference.infer_buildings import infer_buildings
from inference.infer_crops import infer_crops

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    extract_water(year)
    preprocess_water(year)
    infer_water(year)
    extract_buildings(year)
    preprocess_buildings(year)
    infer_buildings(year)
    extract_crops(year)
    preprocess_crops(year)
    infer_crops(year)