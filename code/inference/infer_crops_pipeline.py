import sys
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

def infer_crops_pipeline(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_crops_probabilities_{year}.csv')
    standalone_model = xgb.XGBClassifier()
    standalone_model.load_model('../models/crops.json')
    standalone_label_encoder = joblib.load('../label_encoders/crop_label_encoder.joblib')
    pipeline_model = xgb.XGBClassifier()
    pipeline_model.load_model('../models/crops_pipeline.json')
    pipeline_label_encoder = joblib.load('../label_encoders/crop_pipeline_label_encoder.joblib')
    crop_features = [
        'ndvi_oct',
        'evi_oct',
        'ndwi_oct',
        'mtci_oct',
        'swir_long_oct',
        'ndvi_nov',
        'evi_nov',
        'ndwi_nov',
        'mtci_nov',
        'swir_long_nov',
        'ndvi_dec',
        'evi_dec',
        'ndwi_dec',
        'mtci_dec',
        'swir_long_dec'
    ]
    pipeline_features = crop_features + [
        'water_probability',
        'building_probability'
    ]
    standalone_prediction = standalone_model.predict(df[crop_features])
    standalone_prediction = standalone_label_encoder.inverse_transform(standalone_prediction)
    pipeline_prediction = pipeline_model.predict(df[pipeline_features])
    pipeline_prediction = pipeline_label_encoder.inverse_transform(pipeline_prediction)
    crop_classes = [
        2101,
        2204,
        2205,
        2302,
        2303,
        2403,
        2404,
        2405,
        2407,
        2413,
        2416,
        2419,
        2420,
        9999
    ]
    target_names = [
        'rice',
        'cassava',
        'pineapple',
        'rubber',
        'oil_palm',
        'durian',
        'rambutan',
        'coconut',
        'mango',
        'longan',
        'jackfruit',
        'mangosteen',
        'longkong',
        'others'
    ]
    crop_true = df['class'].where(
        df['class'].isin(crop_classes),
        9999
    )

    for name, prediction in [
        (
            'Standalone crop model',
            standalone_prediction
        ),
        (
            'Pipeline crop model',
            pipeline_prediction
        )
    ]:
        print(name)
        print(classification_report(
            crop_true,
            prediction,
            labels=crop_classes,
            target_names=target_names,
            zero_division=0
        ))

    df['standalone_prediction'] = standalone_prediction
    df['pipeline_prediction'] = pipeline_prediction
    df.to_csv(
        f'../datasets/final_classification_probabilities_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    infer_crops_pipeline(year)
