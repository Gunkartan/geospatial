import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.features import UPSTREAM_MODELS, UPSTREAM_MONTHLY_FEATURES

def infer_water(year: int) -> None:
    df = pd.read_csv(
        f'../datasets/preprocessed_water_final_pipeline_{year}.csv',
        dtype={feature: 'float32' for feature in UPSTREAM_MONTHLY_FEATURES}
    )
    probabilities = df[[
        'row',
        'col'
    ]].copy()

    for name, (model_file, features) in UPSTREAM_MODELS.items():
        model = xgb.XGBClassifier()
        model.load_model(f'../models/{model_file}.json')
        probabilities[f'{name}_probability'] = model.predict_proba(df[features])[
            :,
            1
        ]

    water_prediction = (probabilities['water_probability'] >= 0.52).astype(int)
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

    if year == 2018:
        # The upstream models were trained on 2018, so the crop model gets
        # out-of-fold probabilities there, matching how 2020 and 2024 look.
        for name in UPSTREAM_MODELS:
            out_of_fold = pd.read_csv(f'../datasets/{name}_out_of_fold_probabilities_2018.csv')
            out_of_fold = probabilities[['row', 'col']].merge(
                out_of_fold,
                on=[
                    'row',
                    'col'
                ],
                how='left'
            )
            probabilities[f'{name}_probability'] = out_of_fold[f'{name}_probability'].to_numpy()

    probabilities.to_csv(
        f'../datasets/upstream_probabilities_{year}.csv',
        index=False
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Please provide a year')

    year = int(sys.argv[1])
    infer_water(year)
