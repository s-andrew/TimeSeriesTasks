from collections.abc import Iterable as IterableInst
from typing import Iterable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lags: Union[int, Iterable[int]]=1, feature_names: Iterable[str]=None):
        if isinstance(lags, int):
            self.lags = tuple(range(1, lags + 1))
        elif isinstance(lags, IterableInst):
            self.lags = tuple(lags)
        else:
            raise TypeError('lags must be int or iterable')

        if feature_names is not None:
            self.feature_names = ['L{}.{}'.format(i, fn) for i in self.lags for fn in feature_names]
        else:
            self.feature_names = None

    def get_n_lags(self):
        return len(self.lags)



    def transform(self, X, y=None, copy=False):
        n_lags = self.get_n_lags()
        if X.shape[0] <= n_lags:
            raise ValueError('n_sample <= lags')
        shape = X.shape[0] - n_lags, X.shape[1] * n_lags
        new_X = np.zeros(shape, dtype='float64')

        for i in range(X.shape[0] - n_lags):
            for lag_i, lag in enumerate(self.lags):
                for feature in range(X.shape[1]):
                    # print(i, X.shape[1] * lag_i + feature, '<=', i + n_lags - lag, feature)
                    if i + n_lags - lag >= 0:
                        new_X[i, X.shape[1] * lag_i + feature] = X[i + n_lags - lag, feature]
                    else:
                        new_X[i, X.shape[1] * lag_i + feature] = np.nan
        return new_X[~np.isnan(new_X).any(axis=1)]

    def fit(self, X=None, y=None, copy=False):
        pass

    def fit_transform(self, X=None, y=None, copy=False):
        return self.transform(X, y, copy)


if __name__ == '__main__':
    LAGS = 3
    offset = LAGS if isinstance(LAGS, int) else max(LAGS)
    series = pd.DataFrame(np.random.normal(size=10), columns=['X'])
    lagger = LagTransformer(lags=LAGS, feature_names=['X'])
    x = lagger.transform(series.values)
    x = pd.DataFrame(x, columns=lagger.feature_names, index=series.index[offset:])
    series[[*lagger.feature_names]] = x
    print(series)