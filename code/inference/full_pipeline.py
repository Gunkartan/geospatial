import sys
from extraction.extract_crops_final_pipeline import extract_crops
from preprocessing.preprocess_crops_final_pipeline import preprocess_crops
from preprocessing.add_probabilities import add_probabilities
from inference.infer_crops_pipeline import infer_crops_pipeline

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Please provide values for all the required arguments')

    year = int(sys.argv[1])
    sampling = sys.argv[2].lower() == 'true'
    extract_crops(
        year=year,
        remove_water_buildings=False,
        sampling=sampling
    )
    preprocess_crops(
        year=year,
        standalone=True
    )
    add_probabilities(year)
    infer_crops_pipeline(year)
