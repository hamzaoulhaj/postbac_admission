import os

import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit


problem_title = 'Prediction of the pct of female admitted in a formation'

Predictions = rw.prediction_types.make_regression()

workflow = rw.workflows.EstimatorExternalData()


score_types = [
    rw.score_types.Accuracy(name='acc'),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)

_target_column_name = 'pct_admitted_female_among_admitted'
_ignore_column_names = ['school_name']

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)