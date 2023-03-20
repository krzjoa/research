#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:45:51 2023

@author: krzysztof
"""

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

from champagne import get_champagne


START_YEAR = 2018
END_YEAR = 2023

SPLIT_DATE = '2022-12-31'

TARGET = ['amount']
FEATURES = ['monthday', 
            'month',
            'weekday',
            'special_event_31_12']

MODELS = {
    'lightgbm': LGBMRegressor(),
    'xgboost': XGBRegressor(),
    'catboost': CatBoostRegressor(),
    
    'lightgbm': LGBMRegressor(),
    
    'linear': LinearRegression()    
}

# =============================================================================
#                              PREPARE THE DATA
# =============================================================================

data = get_champagne(years=(START_YEAR, END_YEAR))

train_data = data.query('date <= @SPLIT_DATE')
test_data = data.query('date > @SPLIT_DATE')

X_train, y_train = train_data[FEATURES], train_data[TARGET]
X_test, y_test, df_results = test_data[FEATURES], test_data[TARGET], test_data[TARGET + ['date']]

# =============================================================================
#                                   TRAIN
# =============================================================================

for model in MODELS.values():
    model.fit(X_train, y_train)

# =============================================================================
#                                   TEST
# =============================================================================

for name, model in MODELS.items():
    y_hat = model.predict(X_test)
    df_results[name] = y_hat 


print(df_results.query('date == "2023-12-31"'))


# train_data.query("(monthday == 31) & (month == 12)")


# df_results.plot_bokeh(x='date', y='lightgbm', figsize=(1200, 600))
# df_results.plot_bokeh(x='date', y='xgboost', figsize=(1200, 600))
# df_results.plot_bokeh(x='date', y='catboost', figsize=(1200, 600))
# df_results.plot_bokeh(x='date', y='linear', figsize=(1200, 600))

