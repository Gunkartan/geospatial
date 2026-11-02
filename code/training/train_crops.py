import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from utils.features import CROP_FEATURES, PROBABILITY_FEATURES

def train(standalone: bool) -> None:
    label_encoder_name = 'crop_label_encoder_standalone' if standalone else 'crop_label_encoder_pipeline'
    model_name = 'standalone_crops' if standalone else 'pipeline_crops'
    df = pd.read_csv(f'../datasets/preprocessed_crops_final_pipeline_2018.csv')
    x = df[CROP_FEATURES if standalone else CROP_FEATURES + PROBABILITY_FEATURES]
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
        f'../label_encoders/{label_encoder_name}.joblib'
    )
    block_size = 256
    blocks = (df['row'] // block_size).astype(str) + '_' + (df['col'] // block_size).astype(str)
    folds = np.empty(
        len(df),
        dtype=int
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for fold, (_, fold_idx) in enumerate(splitter.split(
        x,
        y_encoded,
        groups=blocks
    )):
        folds[fold_idx] = fold

    x_train, y_train = x[folds >= 2], y_encoded[folds >= 2]
    x_cv, y_cv = x[folds == 1], y_encoded[folds == 1]
    x_test, y_test = x[folds == 0], y_encoded[folds == 0]
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
    model.save_model(f'../models/{model_name}.json')
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
    if len(sys.argv) != 2:
        raise ValueError('Please provide values for all the required arguments')

    standalone = sys.argv[1].lower() == 'true'
    train(standalone)