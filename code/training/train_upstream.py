import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from utils.spatial import spatial_blocks, spatial_folds, spatial_split
from utils.features import MIXED_LAND_USE_LABEL
from utils.dropout import blank_october

TRAINING_PIXEL_CAP = 8_000_000
CV_PIXEL_CAP = 2_670_000

def best_threshold(
    y: np.ndarray,
    probability: np.ndarray
) -> pd.Series | None:
    results = []

    for threshold in np.linspace(
        0,
        1,
        101
    ):
        preds = (probability >= threshold).astype(int)
        results.append([
            threshold,
            precision_score(
                y,
                preds,
                zero_division=0
            ),
            recall_score(
                y,
                preds,
                zero_division=0
            ),
            f1_score(
                y,
                preds,
                zero_division=0
            )
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

    return None if valid.empty else valid.loc[valid['f1'].idxmax()]

def train_upstream(
    name: str,
    model_file: str,
    features: list[str],
    class_prefix: str
) -> None:
    """Train a binary land-use group model (water, building, ...) on every labelled 2018 pixel with 10 km
    spatial blocks, the same scheme as the crop model.

    Fold 0 is the saved model: 60/20/20 train/CV/test blocks, tree count
    early-stopped on CV, threshold picked on CV, scores reported on test.
    All 5 folds also give out-of-fold 2018 probabilities for the crop model, so
    each pixel's probability comes from a model that never saw its block, as
    happens for 2020 and 2024.
    """
    os.makedirs(
        '../models',
        exist_ok=True
    )
    df = pd.read_csv(
        '../datasets/preprocessed_water_final_pipeline_2018.csv',
        usecols=[
            'row',
            'col',
            'class',
            *features
        ],
        dtype={feature: 'float32' for feature in features}
    )
    x = df[features]
    x = x.where(np.isfinite(x))
    df = df[[
        'row',
        'col',
        'class'
    ]]
    rng = np.random.default_rng(42)
    codes = df['class'].astype(int)
    # 32767 is mixed land use clipped by the int16 label raster, not a group-3 code.
    y = (codes.astype(str).str.startswith(class_prefix) & (codes != MIXED_LAND_USE_LABEL)).astype(int).to_numpy()
    groups = spatial_blocks(df)
    out_of_fold = np.full(
        len(df),
        np.nan,
        dtype='float32'
    )

    for fold, (rest_idx, test_idx) in enumerate(spatial_folds(
        y,
        groups,
        5
    )):
        train_pos, cv_pos = spatial_split(
            y[rest_idx],
            groups[rest_idx],
            4
        )
        train_idx = rest_idx[train_pos]
        cv_idx = rest_idx[cv_pos]

        # A random subset keeps training time manageable with 3 months of features;
        # more pixels from the same blocks add little.
        if len(train_idx) > TRAINING_PIXEL_CAP:
            train_idx = np.sort(rng.choice(
                train_idx,
                TRAINING_PIXEL_CAP,
                replace=False
            ))

        if len(cv_idx) > CV_PIXEL_CAP:
            cv_idx = np.sort(rng.choice(
                cv_idx,
                CV_PIXEL_CAP,
                replace=False
            ))

        model = xgb.XGBClassifier(
            n_estimators=800,
            learning_rate=0.2,
            max_depth=7,
            subsample=0.7,
            colsample_bytree=1.0,
            random_state=42,
            scale_pos_weight=2,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric='logloss'
        )
        # Pixels with missing features are kept, and October is blanked for a random
        # share of train and CV pixels, so the model also works where October is missing.
        model.fit(
            blank_october(
                x.iloc[train_idx],
                rng
            ),
            y[train_idx],
            eval_set=[(
                blank_october(
                    x.iloc[cv_idx],
                    rng
                ),
                y[cv_idx]
            )],
            verbose=False
        )
        out_of_fold[test_idx] = model.predict_proba(x.iloc[test_idx])[
            :,
            1
        ]
        scored = test_idx
        print(f'Fold {fold}: {model.best_iteration + 1} trees, '
              f'test AUC {roc_auc_score(y[scored], out_of_fold[scored]):.3f}', flush=True)

        if fold == 0:
            model.save_model(f'../models/{model_file}.json')
            cv_probability = model.predict_proba(x.iloc[cv_idx])[
                :,
                1
            ]
            best = best_threshold(
                y[cv_idx],
                cv_probability
            )

            if best is None:
                print('There are no thresholds with a precision or recall of more than 0.8')

            else:
                test_prediction = (out_of_fold[scored] >= best['thresholds']).astype(int)
                print(f'The best threshold on CV is {best["thresholds"]:.3f}')
                print(f'On test blocks, the precision is {precision_score(y[scored], test_prediction):.3f}')
                print(f'On test blocks, the recall is {recall_score(y[scored], test_prediction):.3f}')
                print(f'On test blocks, the F1 score is {f1_score(y[scored], test_prediction):.3f}')

    print(f'Out-of-fold AUC over all 2018 pixels: {roc_auc_score(y, out_of_fold):.3f}')
    probabilities = df[[
        'row',
        'col'
    ]].copy()
    probabilities[f'{name}_probability'] = out_of_fold
    probabilities.to_csv(
        f'../datasets/{name}_out_of_fold_probabilities_2018.csv',
        index=False
    )
