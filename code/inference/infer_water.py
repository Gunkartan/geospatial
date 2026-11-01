import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.features import WATER_FEATURES, BUILDING_FEATURES

def infer_water(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_water_final_pipeline_{year}.csv')
    water_model = xgb.XGBClassifier()
    water_model.load_model('../models/water.json')
    building_model = xgb.XGBClassifier()
    building_model.load_model('../models/buildings.json')
    water_probability = water_model.predict_proba(df[WATER_FEATURES])[
        :,
        1
    ]
    building_probability = building_model.predict_proba(df[BUILDING_FEATURES])[
        :,
        1
    ]
    water_prediction = (water_probability >= 0.52).astype(int)
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