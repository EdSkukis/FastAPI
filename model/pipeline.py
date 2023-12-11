import json

import dill
import pandas as pd
import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(df):
    df_temp = df.copy()
    columns_to_drop = [
       'id',
       'url',
       'region',
       'region_url',
       'price',
       'manufacturer',
       'image_url',
       'description',
       'posting_date',
       'lat',
       'long'
    ]
    return df_temp.drop(columns_to_drop, axis=1)


def outlier_removal(df):
    name = 'year'
    df_temp = df.copy()
    q1 = df_temp[name].quantile(0.25)
    q3 = df_temp[name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return df_temp[(df_temp[name] >= lower_bound) & (df_temp[name] <= upper_bound)]


def short_model(x):
    if not pd.isna(x):
        return x.lower().split(' ')[0]
    else:
        return x


def get_short_model(df):
    df_temp = df.copy()
    df_temp.loc[:, 'short_model'] = df_temp['model'].apply(short_model)
    return df_temp


def get_age_category(df):
    df_temp = df.copy()
    df_temp.loc[:, 'age_category'] = df_temp['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df_temp


def main():
    file_patch = r'data/homework.csv'
    df = pd.read_csv(file_patch)

    preprocessor_0 = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('removal', FunctionTransformer(outlier_removal)),
        ('short_model', FunctionTransformer(get_short_model)),
        ('age_category', FunctionTransformer(get_age_category))
    ])

    # df = preprocessor_0.fit_transform(df)

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        # RandomForestClassifier(),
        # SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            # ('preprocessor_0', preprocessor_0.fit_transform),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)


    metadata = {
        'name': 'Car price prediction model',
        'author': 'Eduards Skukis',
        'version': 1,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'type': type(best_pipe.named_steps["classifier"]).__name__,
        'accuracy': best_score
    }

    with open('pipeline.pkl', 'wb') as file:
        dill.dump({'model': best_pipe, 'metadata': metadata}, file, recurse=True)

    with open(r'data/7310993818.json') as file:
        X = json.load(file)
    df = pd.DataFrame.from_dict([X])

    y = best_pipe.predict(df)
    print(y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
