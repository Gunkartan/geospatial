import sys
import joblib
import pandas as pd
import xgboost as xgb

def inference(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_all_{year}.csv')
    water_model = xgb.XGBClassifier()
    water_model.load_model('../models/water.json')
    building_model = xgb.XGBClassifier()
    building_model.load_model('../models/buildings.json')
    crop_model = xgb.XGBClassifier()
    crop_model.load_model('../models/crops.json')
    crop_label_encoder = joblib.load('../label_encoders/crop_label_encoder.joblib')
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
    building_features = water_features
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
    water_df = df[water_features]
    water_probability = water_model.predict_proba(water_df)[
        :,
        1
    ]
    df = df[water_probability < 0.56].copy()
    building_df = df[building_features]
    building_probability = building_model.predict_proba(building_df)[
        :,
        1
    ]
    df = df[building_probability < 0.56].copy()
    crop_df = df[crop_features]
    crop_prediction = crop_model.predict(crop_df)
    df['class'] = crop_label_encoder.inverse_transform(crop_prediction)
    df.to_csv(
        f'../datasets/final_classification_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    inference(year)