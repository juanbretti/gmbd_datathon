# -*- coding: utf-8 -*-

# %%
## Libraries ----
# General usage
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump, load

# Custom library
import helpers

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
## Read dataframes ----
df_casos_uci = load('storage/df_export_cases_uci_lm.joblib')
df_casos = load('storage/df_export_cases_lm.joblib')
df_aemet = load('storage/df_export_aemet_lm.joblib')
df_googletrends = load('storage/df_export_googletrends_lm.joblib')
df_mitma = load('storage/df_export_mitma_lm.joblib')
df_mscbs = load('storage/df_export_mscbs_lm.joblib')
df_holidays = load('storage/df_export_holidays_lm.joblib')

# %%
## Constants ----
TARGET_VARIABLE = 'uci_target__ratio_population__num_def'
PREDICTION_WINDOW = 7
LAG_UCI = [1]
LAG_CASOS = [1]
LAG_OTHER = [0]
MAX_LAG = max(LAG_UCI+LAG_CASOS+LAG_OTHER)
SEED = 42

# %%
## Prepare dataframes ----
fix_columns = ['fecha', 'Code provincia alpha', 'Code comunidad autónoma alpha']

def add_prefix(df, prefix, exclude=fix_columns):
    columns_all = pd.Series(df.columns)
    columns_to_prefix = pd.Series(columns_all).isin(exclude)
    columns_all[~columns_to_prefix] = prefix+columns_all[~columns_to_prefix]
    df.columns = columns_all
    return df

### Target variable ----
df_casos_uci_target = df_casos_uci[fix_columns+['ratio_population__num_def']]
df_casos_uci_target = add_prefix(df_casos_uci_target, 'uci_target__')
### Death age groups ----
df_casos_uci_lagged = helpers.shift_timeseries_by_lags(df_casos_uci, fix_columns=fix_columns, lag_numbers=LAG_UCI)
df_casos_uci_lagged = add_prefix(df_casos_uci_lagged, 'uci__')
### Cases tested ----
df_casos_lagged = helpers.shift_timeseries_by_lags(df_casos, fix_columns=fix_columns, lag_numbers=LAG_CASOS)
df_casos_lagged = add_prefix(df_casos_lagged, 'tests__')
### AEMET temperature ----
df_aemet_lagged = helpers.shift_timeseries_by_lags(df_aemet, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_aemet_lagged = add_prefix(df_aemet_lagged, 'aemet__')
### Google Trends ----
df_googletrends_lagged = helpers.shift_timeseries_by_lags(df_googletrends, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_googletrends_lagged = add_prefix(df_googletrends_lagged, 'google_trends__')
### Movements to Madrid (`mitma`) ----
df_mitma_lagged = helpers.shift_timeseries_by_lags(df_mitma, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_mitma_lagged = add_prefix(df_mitma_lagged, 'mitma__')
### Vaccination in Spain (`mscbs`) ----
df_mscbs_lagged = helpers.shift_timeseries_by_lags(df_mscbs, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_mscbs_lagged = add_prefix(df_mscbs_lagged, 'mscbs__')
### Holidays ----
df_holidays_lagged = helpers.shift_timeseries_by_lags(df_holidays, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_holidays_lagged = add_prefix(df_holidays_lagged, 'holidays__')

# %%
## Merge all datasources ----
df_merge = df_casos_uci_target.copy()
df_merge = df_merge.merge(df_casos_uci_lagged, on=fix_columns)
df_merge = df_merge.merge(df_casos_lagged, on=fix_columns)
df_merge = df_merge.merge(df_aemet_lagged, on=fix_columns)
df_merge = df_merge.merge(df_googletrends_lagged, on=fix_columns)
df_merge = df_merge.merge(df_mitma_lagged, on=fix_columns)
df_merge = df_merge.merge(df_mscbs_lagged, on=fix_columns)
df_merge = df_merge.merge(df_holidays_lagged, on=fix_columns)

df_merge = df_merge.fillna(df_merge.mean(numeric_only=True))

# %%
## Feature engineering ----
### Filter only vaccination times

### Dates ----
# TODO: Fix `bisiesto`
df_merge['fecha_year'] = df_merge['fecha'].dt.year
df_merge['fecha_month'] = df_merge['fecha'].dt.month
df_merge['fecha_day'] = df_merge['fecha'].dt.day
df_merge['fecha_dayofweek'] = df_merge['fecha'].dt.dayofweek
df_merge['fecha_dayofyear'] = df_merge['fecha'].dt.dayofyear
df_merge['fecha_weekend'] = [(x in [5,6])*1 for x in df_merge['fecha_dayofweek']]
df_merge_ts = df_merge.copy()  # For the `time series`
df_merge = df_merge.drop(columns=['fecha'])

# %%
## Split dataset ----
# Calculate prediction window
split_ = df_merge.shape[0]-PREDICTION_WINDOW

# Split
df_train = df_merge.iloc[MAX_LAG:split_, :]
df_test = df_merge.iloc[split_:(split_+PREDICTION_WINDOW), :]

X_train = df_train.drop(columns=TARGET_VARIABLE)
X_test = df_test.drop(columns=TARGET_VARIABLE)
y_train = df_train[TARGET_VARIABLE]
y_test = df_test[TARGET_VARIABLE]

# %%
# Linear model
import statsmodels.api as sm

def feature_coefficient(X, y, threshold_coef=0.01, threshold_pvalue=0.05):
    """List the significant features

    Args:
        X (DataFrame): Features data frame
        y (Series): Target variable
        threshold_coef (float, optional): Threshold value for the coefficient. Defaults to 0.01.
        threshold_pvalue (float, optional): Theshold value for the p-value. Defaults to 0.05.

    Returns:
        DataFrame: Table with the significant features
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary()) #Summary of the model

    summary_ = pd.DataFrame({'Feature': results.pvalues.keys(), 'Coefficient': results.params.values, 'p-value': results.pvalues.values})
    filter_ = (summary_['Coefficient'].abs() > threshold_coef) & (summary_['p-value'].abs() < threshold_pvalue)
    summary_ = summary_[filter_].sort_values('Coefficient', ascending=False)
    return summary_[['Feature', 'Coefficient']]

X_train_dummies = pd.get_dummies(X_train, columns = ['Code provincia alpha'], drop_first=True)
X_train_dummies = X_train_dummies.drop(columns=fix_columns, errors='ignore')

# List of the most relevant features/columns of the dataset.
print('List the significant features')
df = feature_coefficient(X_train_dummies, y_train)
df.style.hide_index()

df.plot.barh(x='Feature', y='Coefficient', xlim=[-3,3], figsize=(15,15))

# %%
