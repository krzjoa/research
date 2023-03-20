#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:54:36 2023

@author: krzysztof
"""

import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import pandas_bokeh

import pdb


def get_champagne(years: Tuple[int, int] = (2018, 2023),
                  normal_days: dict =  {'n': 1000, 'p': 0.02},
                  weekdays: dict = {4: {'n': 1800, 'p': 0.03},
                                    5: {'n': 2000, 'p': 0.05}},
                  special_events: dict = {(29, 12): {'n': 1500, 'p': 0.1},
                                          (30, 12): {'n': 2000, 'p': 0.2},
                                          (31, 12): {'n': 2000, 'p': 0.7}}):
    """Pytanie: być może to powinno być zamodelowane w inny sposób
       W tej chwili n to nie liczba klientów, tylko liczba sytuacji, kiedy klient
       zastanawia się nad kupnem. Pytanie brzmi, czy jeśli wprowadzimy dodatkowy rozkład
       z dodatkowym parametrem, czy ma to jakieś znaczenie?
       Tzn. n=liczba_klientów, p=prawdopodobieństwo_zakupu, b=liczba_sztuk
       
       Z drugiej strony, taki rozkład może być tak czy inaczej, łatwiejszy do 
       interpretacji
       
    """
    
    df = pd.DataFrame({
        'date': pd.date_range(f'{years[0]}-01-01', f'{years[1]}-12-31')    
    })
    df['weekday'] = df.date.dt.weekday
    df['month'] = df.date.dt.month
    df['monthday'] = df.date.dt.day
    
    normal_days_amount = np.random.binomial(**normal_days, size=len(df))
    df['amount'] = normal_days_amount
    
    if weekdays:
        for wd, params in weekdays.items():
            mask = df.weekday == wd
            df.loc[mask, 'amount'] = np.random.binomial(**params, size=mask.sum())
    
    if special_events:
        for d, params in special_events.items():
            mask = (df.monthday == d[0]) & (df.month == d[1]) 
            df.loc[mask, 'amount'] = np.random.binomial(**params, size=mask.sum())
            col_name = f'special_event_{d[0]}_{d[1]}'
            df[col_name] = 0
            df.loc[mask, col_name] = 1
    
    return df
    



if __name__ == '__main__':
    df = get_champagne()
    
    df.plot_bokeh(x='date', y='amount', figsize=(1200, 600))
    
    df \
        .query('"2023-01-01" < date < "2023-01-31"') \
        .plot_bokeh(x='date', y='amount', figsize=(1200, 600))
