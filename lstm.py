import math
import time
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from LagTransformer import LagTransformer


if __name__ == '__main__':
    LOOK_BACK = 1
    EPOCH = 6
    N_FOLDS = 10

    df = pd.read_csv('data/test.csv')

    X = df.values.astype('float64')
    y = df[['y1']].values.astype('float64')

    n_features = X.shape[1]
    results = []
    Result = namedtuple('Result', 'index predict score')

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    for train_index, test_index in tscv.split(X):
        start_time = time.time()
        # Preprocessing
        features_preprocessor = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(0, 1))),
            ('lag_transformer', LagTransformer(lags=LOOK_BACK)),
            ('reshaper', FunctionTransformer(lambda X: np.reshape(X, (X.shape[0], 1, X.shape[1]))))
        ])
        scaler = MinMaxScaler(feature_range=(0, 1))
        target_preprocessor = Pipeline([
            ('scaler', scaler),
            ('reshaper', FunctionTransformer(lambda X: X.reshape((X.shape[0]), )[LOOK_BACK:]))
        ])

        # LSTM model
        model = Sequential()
        model.add(LSTM(256, input_shape=(1, n_features * LOOK_BACK)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Fit
        if not os.path.exists('./graph'):
            os.makedirs('./graph')
        tb_callback = TensorBoard('./graph', histogram_freq=0, write_graph=True, write_images=True)

        X_train = features_preprocessor.fit_transform(X[train_index])
        y_train = target_preprocessor.fit_transform(y[train_index])
        model.fit(X_train, y_train, epochs=EPOCH, batch_size=1, verbose=10, callbacks=[tb_callback])

        # Tests
        X_test = features_preprocessor.transform(X[test_index])
        y_test = target_preprocessor.transform(y[test_index])

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

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
            'train': Result(train_index, train_index, train_score),
            'test': Result(test_index, test_pred, test_score)
        })

    mean_train_score = sum(result['train'].score for result in results) / len(results)
    max_train_score = max(result['train'].score for result in results)
    print('TRAIN SCORE: {:.2f}(max) {:.2f}(mean)'.format(max_train_score, mean_train_score))

    mean_test_score = sum(result['test'].score for result in results) / len(results)
    max_test_score = max(result['test'].score for result in results)
    print('TEST SCORE: {:.2f}(max) {:.2f}(mean)'.format(max_test_score, mean_test_score))

    for i, result in enumerate(results):
        plt.plot(y, '#aacdff', label='origin')
        plt.plot(result['train'].index[LOOK_BACK:], result['train'].predict.reshape(-1),
                 label='train prediction (RMSE={0:.2f})'.format(result['train'].score))
        plt.plot(result['test'].index[LOOK_BACK:], result['test'].predict.reshape(-1),
                 label='test prediction (RMSE={0:.2f})'.format(result['test'].score))
        plt.legend()
        plt.show()

