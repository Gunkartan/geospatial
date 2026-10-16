import sys
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

def infer_crops(
    year: int,
    standalone: bool
) -> None:
    filename = f'preprocessed_crops_standalone_{year}' if standalone else f'preprocessed_crops_final_pipeline_{year}'
    output_name = f'final_classification_standalone_{year}' if standalone else f'final_classification_pipeline_{year}'
    df = pd.read_csv(f'../datasets/{filename}.csv')
    crop_model = xgb.XGBClassifier()
    crop_model.load_model('../models/crops.json')
    crop_label_encoder = joblib.load('../label_encoders/crop_label_encoder.joblib')
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
    crop_prediction = crop_model.predict(df[crop_features])
    crop_prediction = crop_label_encoder.inverse_transform(crop_prediction)
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
    evaluation_true = crop_true.tolist()
    evaluation_prediction = crop_prediction.tolist()

    if not standalone:
        excluded_file = f'../datasets/raw_crops_excluded_{year}.csv'
        excluded_df = pd.read_csv(excluded_file)
        excluded_true = excluded_df['class'].where(
            excluded_df['class'].isin(crop_classes),
            9999
        )
        evaluation_true.extend(excluded_true.tolist())
        evaluation_prediction.extend([9999] * len(excluded_df))

    print('Crop model')
    print(classification_report(
        evaluation_true,
        evaluation_prediction,
        labels=crop_classes,
        target_names=target_names,
        zero_division=0
    ))
    df['class'] = crop_prediction

    if not standalone:
        excluded_df['class'] = 9999
        excluded_df = excluded_df.reindex(columns=df.columns)
        df = pd.concat(
            [
                df,
                excluded_df
            ],
            ignore_index=True
        )
        
    df.to_csv(
        f'../datasets/{output_name}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Please provide values for all the required arguments')

    year = int(sys.argv[1])
    standalone = sys.argv[2].lower() == 'true'
    infer_crops(
        year=year,
        standalone=standalone
    )