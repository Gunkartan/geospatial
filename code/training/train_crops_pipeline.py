import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from extraction.extract_crops_final_pipeline import extract_crops
from preprocessing.preprocess_crops_final_pipeline import preprocess_crops
from preprocessing.add_probabilities import add_probabilities

def train_crops_pipeline(year: int) -> None:
    df = pd.read_csv(f'../datasets/preprocessed_crops_probabilities_{year}.csv')
    x = df.drop(columns=[
        'row',
        'col',
        'class'
    ])
    y = df['class']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    os.makedirs(
        '../models',
        exist_ok=True
    )
    os.makedirs(
        '../label_encoders',
        exist_ok=True
    )
    joblib.dump(
        le,
        '../label_encoders/crop_pipeline_label_encoder.joblib'
    )
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y_encoded,
        test_size=0.4,
        random_state=42,
        stratify=y_encoded
    )
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.1,
        max_depth=10,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        x_train,
        y_train
    )
    model.save_model('../models/crops_pipeline.json')
    y_cv_pred = model.predict(x_cv)
    class_map = {
        2101: 'rice',
        2204: 'cassava',
        2205: 'pineapple',
        2302: 'rubber',
        2303: 'oil_palm',
        2403: 'durian',
        2404: 'rambutan',
        2405: 'coconut',
        2407: 'mango',
        2413: 'longan',
        2416: 'jackfruit',
        2419: 'mangosteen',
        2420: 'longkong',
        9999: 'others'
    }
    class_names = [class_map[class_id] for class_id in le.classes_]
    print(classification_report(
        y_cv,
        y_cv_pred,
        labels=np.arange(len(le.classes_)),
        target_names=class_names,
        zero_division=0
    ))
    print(f'The precision is {precision_score(
        y_cv,
        y_cv_pred,
        average='weighted'
    ):.3f}')
    print(f'The recall is {recall_score(
        y_cv,
        y_cv_pred,
        average='weighted'
    ):.3f}')
    print(f'The F1 score is {f1_score(
        y_cv,
        y_cv_pred,
        average='weighted'
    ):.3f}')

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
    train_crops_pipeline(year)
