import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

def infer_water(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_water_final_pipeline_{year}.csv')
    water_model = xgb.XGBClassifier()
    water_model.load_model('../models/water.json')
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
    probability = water_model.predict_proba(df[water_features])[
        :,
        1
    ]
    water_prediction = (probability >= 0.56).astype(int)
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
    water_indices = df.loc[
        probability >= 0.56,
        [
            'row',
            'col'
        ]
    ]
    water_indices.to_csv(
        f'../datasets/water_indices_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    infer_water(year)