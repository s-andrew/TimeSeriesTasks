import math
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def get_naive_predict(series, lag=1):
    return np.concatenate([[np.nan] * lag, series[:-lag]]).T


if __name__ == '__main__':
    LOOK_BACK = 1
    N_FOLDS = 10

    df = pd.read_csv('data/test.csv')

    X = df.values.astype('float64')
    y = df[['y1']].values.astype('float64')

    n_features = X.shape[1]
    results = []
    Result = namedtuple('Result', 'index predict score')

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    for train_index, test_index in tscv.split(X):
        y_train, y_test = y[train_index], y[test_index]
        train_pred = get_naive_predict(y_train.reshape(-1), LOOK_BACK)
        test_pred = get_naive_predict(y_test.reshape(-1), LOOK_BACK)

        train_score = math.sqrt(mean_squared_error(y_train[LOOK_BACK:], train_pred[LOOK_BACK:]))
        print('Train Score: {:.2f} RMSE'.format(train_score))
        test_score = math.sqrt(mean_squared_error(y_test[LOOK_BACK:], test_pred[LOOK_BACK:]))
        print('Test Score: %.2f RMSE' % test_score)

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



