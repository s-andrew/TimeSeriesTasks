import time
import math
from collections import namedtuple

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from catboost import CatBoostRegressor

from LagTransformer import LagTransformer


if __name__ == '__main__':
    LOOK_BACK = 4
    N_FOLDS = 10

    df = pd.read_csv('data/test.csv')

    X = df.values.astype('float64')
    y = df[['y1']].values.astype('float64')

    results = []
    Result = namedtuple('Result', 'index predict score')

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    for train_index, test_index in tscv.split(X):
        start_time = time.time()
        # Preprocessing
        lag_transformer = LagTransformer(lags=LOOK_BACK, feature_names=df.columns)
        features_preprocessor = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(0, 1))),
            ('lag_transformer', lag_transformer)
        ])
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_preprocessor = Pipeline([
            ('scaler', scaler),
            ('reshaper', FunctionTransformer(lambda X: X.reshape((X.shape[0]), )[LOOK_BACK:]))
        ])

        cat_origin = {'s1', 's2'}
        cat_indeces = [i for i, name in enumerate(lag_transformer.feature_names) if name.split('.')[1] in cat_origin]

        # Catboost model
        model = CatBoostRegressor(learning_rate=0.03,
                                  n_estimators=1000,
                                  depth=6,
                                  cat_features=cat_indeces,
                                  task_type='GPU',
                                  logging_level='Silent')

        # Fit
        X_train = features_preprocessor.fit_transform(X[train_index])
        y_train = target_preprocessor.fit_transform(y[train_index])
        model.fit(X_train, y_train)

        # Tests
        X_test = features_preprocessor.transform(X[test_index])
        y_test = target_preprocessor.transform(y[test_index])

        train_pred = model.predict(X_train).reshape(-1, 1)
        test_pred = model.predict(X_test).reshape(-1, 1)

        train_pred = scaler.inverse_transform(train_pred)
        test_pred = scaler.inverse_transform(test_pred)
        y_train = scaler.inverse_transform([y_train])
        y_test = scaler.inverse_transform([y_test])


        train_score = math.sqrt(mean_squared_error(y_train[0], train_pred[:, 0]))
        print('Train Score: {:.2f} RMSE'.format(train_score))
        test_score = math.sqrt(mean_squared_error(y_test[0], test_pred[:, 0]))
        print('Test Score: {:.2f} RMSE'.format(test_score))
        print('Train time: {:.2f}s'.format(time.time() - start_time))
        print()

        results.append({
            'train': Result(train_index, train_pred, train_score),
            'test': Result(test_index, test_pred, test_score)
        })



    mean_train_score = sum(result['train'].score for result in results) / len(results)
    max_train_score = max(result['train'].score for result in results)
    print('TRAIN SCORE: {:.2f}(max) {:.2f}(mean)'.format(max_train_score, mean_train_score))

    mean_test_score = sum(result['test'].score for result in results) / len(results)
    max_test_score = max(result['test'].score for result in results)
    print('TEST SCORE: {:.2f}(max) {:.2f}(mean)'.format(max_test_score, mean_test_score))

    plt.title('Catboost Regressor validation on {} folds'.format(N_FOLDS))
    plt.plot(df.y1, '#aacdff', label='origin')
    for i, result in enumerate(results):
        plt.plot(result['test'].index[LOOK_BACK:], result['test'].predict.reshape(-1),
                 label='test prediction on {} fold (RMSE={:.2f})'.format(str(i + 1), result['test'].score))
    plt.legend()
    plt.show()



