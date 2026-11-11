import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from utils.features import PROBABILITY_COLUMNS
from utils.spatial import spatial_blocks, spatial_split
from utils.dropout import blank_october

def train(standalone: bool) -> None:
    label_encoder_name = 'crop_label_encoder_standalone' if standalone else 'crop_label_encoder_pipeline'
    model_name = 'standalone_crops' if standalone else 'pipeline_crops'
    df = pd.read_csv(f'../datasets/preprocessed_crops_final_pipeline_2018.csv')
    x = df.drop(columns=[
        'row',
        'col',
        'class',
        *(PROBABILITY_COLUMNS if standalone else [])
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
        f'../label_encoders/{label_encoder_name}.joblib'
    )
    groups = spatial_blocks(df)
    # 1/5 of blocks held out for test, then 1/4 of the rest for CV: ~60/20/20.
    rest_idx, test_idx = spatial_split(
        y_encoded,
        groups,
        5
    )
    train_pos, cv_pos = spatial_split(
        y_encoded[rest_idx],
        groups[rest_idx],
        4
    )
    train_idx = rest_idx[train_pos]
    cv_idx = rest_idx[cv_pos]
    # October is blanked for a random share of train and CV rows, so the model learns
    # a Nov/Dec fallback for pixels whose October scene is missing (half of 2020).
    rng = np.random.default_rng(42)
    x_train, y_train = blank_october(x.iloc[train_idx], rng), y_encoded[train_idx]
    x_cv, y_cv = blank_october(x.iloc[cv_idx], rng), y_encoded[cv_idx]
    x_test, y_test = x.iloc[test_idx], y_encoded[test_idx]
    print(f'Spatial split: {len(train_idx)} train / {len(cv_idx)} CV / {len(test_idx)} test pixels, '
          f'{len(np.unique(groups[train_idx]))}/{len(np.unique(groups[cv_idx]))}/{len(np.unique(groups[test_idx]))} blocks')
    # Depth sweeps under the spatial split scored the same at depth 2-10 once the
    # tree count was early-stopped on the CV blocks; deeper, unstopped models only
    # memorize 2018 parcels. CV picks the tree count, so report on test blocks.
    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric='mlogloss'
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(
            x_cv,
            y_cv
        )],
        verbose=False
    )
    print(f'Early stopping kept {model.best_iteration + 1} trees')
    model.save_model(f'../models/{model_name}.json')
    y_test_pred = model.predict(x_test)
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
        y_test,
        y_test_pred,
        labels=np.arange(len(le.classes_)),
        target_names=class_names,
        zero_division=0
    ))
    print(f'The precision is {precision_score(
        y_test,
        y_test_pred,
        average='weighted'
    ):.3f}')
    print(f'The recall is {recall_score(
        y_test,
        y_test_pred,
        average='weighted'
    ):.3f}')
    print(f'The F1 score is {f1_score(
        y_test,
        y_test_pred,
        average='weighted'
    ):.3f}')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide values for all the required arguments')

    standalone = sys.argv[1].lower() == 'true'
    train(standalone)