import os

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, \
    OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def get_estimator():

    cat_cols = [
        'school_status','category_school','path','year','department','region_name'
    ]
    drop_cols = [
        "super_path","sub_path","nb_applicants_female","nb_admitted_female"
    ]

    categorical_transformer = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('drop cols', 'drop', drop_cols),
        ], remainder='passthrough')

    regressor = RandomForestRegressor(
            n_estimators=5, max_depth=50, max_features=10
        )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', regressor)
    ])

    return pipeline