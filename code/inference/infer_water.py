import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

def infer_water(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_water_final_pipeline_{year}.csv')
    water_model = xgb.XGBClassifier()
    water_model.load_model('../models/water.json')
    building_model = xgb.XGBClassifier()
    building_model.load_model('../models/buildings.json')
    water_features = [
        'ndwi_mean',
        'mndwi_mean',
        'bsi_variance',
        'ndti_mean',
        'ndbsi_variance',
        'bi',
        'bi_mean',
        'bi_variance',
        'rbc_mean',
        'rsr_mean',
        'rsr_variance',
        'rgri_mean',
        'snr_mean',
        'snr_variance'
    ]
    building_features = [
        'ndvi',
        'ndwi_mean',
        'ndwi_variance',
        'evi_mean',
        'ndbi_mean',
        'mndwi_mean',
        'mndwi_variance',
        'ndti',
        'ndti_mean',
        'ndbsi_mean',
        'ndbsi_variance',
        'bi',
        'bi_mean',
        'bi_variance',
        'rbc_mean',
        'grri_mean',
        'grri_variance',
        'rsr_mean',
        'rsr_variance',
        'rgri_mean',
        'snr_variance'
    ]
    water_probability = water_model.predict_proba(df[water_features])[
        :,
        1
    ]
    building_probability = building_model.predict_proba(df[building_features])[
        :,
        1
    ]
    water_prediction = (water_probability >= 0.56).astype(int)
    water_true = df['class'].astype(str).str.startswith('4').astype(int)
    p_water = precision_score(
        water_true,
        water_prediction
    )
    r_water = recall_score(
        water_true,
        water_prediction
    )
    f1_water = f1_score(
        water_true,
        water_prediction
    )
    print('Water model')
    print(f'The precision is {p_water:.3f}')
    print(f'The recall is {r_water:.3f}')
    print(f'The F1 score is {f1_water:.3f}')
    probabilities = df[[
        'row',
        'col'
    ]].copy()
    probabilities['water_probability'] = water_probability
    probabilities['building_probability'] = building_probability
    probabilities.to_csv(
        f'../datasets/upstream_probabilities_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    infer_water(year)