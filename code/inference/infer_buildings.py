import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

def infer_buildings(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_buildings_final_pipeline_{year}.csv')
    building_model = xgb.XGBClassifier()
    building_model.load_model('../models/buildings.json')
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
    probability = building_model.predict_proba(df[building_features])[
        :,
        1
    ]
    building_prediction = (probability >= 0.56).astype(int)
    building_true = df['class'].astype(str).str.startswith('1').astype(int)
    p_buildings = precision_score(
        building_true,
        building_prediction
    )
    r_buildings = recall_score(
        building_true,
        building_prediction
    )
    f1_buildings = f1_score(
        building_true,
        building_prediction
    )
    print('Building model')
    print(f'The precision is {p_buildings:.3f}')
    print(f'The recall is {r_buildings:.3f}')
    print(f'The F1 score is {f1_buildings:.3f}')
    building_probabilities = df[[
        'row',
        'col'
    ]].copy()
    building_probabilities['building_probability'] = probability
    building_probabilities.to_csv(
        f'../datasets/building_probabilities_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    infer_buildings(year)