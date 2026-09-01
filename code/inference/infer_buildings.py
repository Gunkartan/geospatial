import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

def infer_buildings(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_buildings_final_pipeline_{year}.csv')
    building_model = xgb.XGBClassifier()
    building_model.load_model('../models/buildings.json')
    water_features = [
        'ndvi',
        'ndvi_mean',
        'ndvi_variance',
        'ndwi',
        'ndwi_mean',
        'ndwi_variance',
        'evi',
        'evi_mean',
        'evi_variance',
        'ndbi',
        'ndbi_mean',
        'ndbi_variance',
        'mndwi',
        'mndwi_mean',
        'mndwi_variance',
        'bsi',
        'bsi_mean',
        'bsi_variance',
        'ndsi',
        'ndsi_mean',
        'ndsi_variance',
        'ndti',
        'ndti_mean',
        'ndti_variance',
        'ndbsi',
        'ndbsi_mean',
        'ndbsi_variance',
        'bi',
        'bi_mean',
        'bi_variance',
        'rbc',
        'rbc_mean',
        'rbc_variance',
        'grri',
        'grri_mean',
        'grri_variance',
        'rsr',
        'rsr_mean',
        'rsr_variance',
        'rgri',
        'rgri_mean',
        'rgri_variance',
        'snr',
        'snr_mean',
        'snr_variance',
        'bri',
        'bri_mean',
        'bri_variance'
    ]
    probability = building_model.predict_proba(df[water_features])[
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
    building_indices = df.loc[
        probability >= 0.56,
        [
            'row',
            'col'
        ]
    ]
    building_indices.to_csv(
        f'../datasets/building_indices_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    infer_buildings(year)