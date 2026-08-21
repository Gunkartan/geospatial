import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == '__main__':
    os.makedirs(
        '../models',
        exist_ok=True
    )
    df = pd.read_csv('../datasets/preprocessed_water.csv')
    x = df.drop(columns=[
        'labels',
        'water'
    ])
    y = df['water']
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.2,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=1.0,
        random_state=42,
        scale_pos_weight=2,
        n_jobs=-1
    )
    model.fit(
        x_train,
        y_train
    )
    model.save_model('../models/water.json')
    y_cv_prob = model.predict_proba(x_cv)[
        :,
        1
    ]
    threshold_list = np.linspace(
        0,
        1,
        101
    )
    results = []

    for threshold in threshold_list:
        preds = (y_cv_prob >= threshold).astype(int)
        p = precision_score(
            y_cv,
            preds,
            zero_division=0
        )
        r = recall_score(
            y_cv,
            preds,
            zero_division=0
        )
        f1 = f1_score(
            y_cv,
            preds,
            zero_division=0
        )
        results.append([
            threshold,
            p,
            r,
            f1
        ])

    metrics = pd.DataFrame(
        results,
        columns=[
            'thresholds',
            'precision',
            'recall',
            'f1'
        ]
    )
    valid = metrics[(metrics['precision'] >= 0.8) | (metrics['recall'] >= 0.8)]

    if not valid.empty:
        best_score = valid.loc[valid['f1'].idxmax()]
        print(f'The best threshold is {best_score["thresholds"]:.3f}')
        print(f'The precision is {best_score["precision"]:.3f}')
        print(f'The recall is {best_score["recall"]:.3f}')
        print(f'The F1 score is {best_score["f1"]:.3f}')

    else:
        print('There are no thresholds with a precision or recall of more than 0.8')