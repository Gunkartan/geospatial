import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    df = pd.read_csv('../datasets/preprocessed_crops.csv')
    x = df.drop(columns=['class'])
    y = df['class']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
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
    y_cv_pred = model.predict(x_cv)
    class_names = [
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
    print(classification_report(
        y_cv,
        y_cv_pred,
        target_names=class_names
    ))
    print(f'The precision is {precision_score(
        y_cv,
        y_cv_pred,
        average='weighted'
    )}')
    print(f'The recall is {recall_score(
        y_cv,
        y_cv_pred,
        average='weighted'
    )}')
    print(f'The F1 score is {f1_score(
        y_cv,
        y_cv_pred,
        average='weighted'
    )}')