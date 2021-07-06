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

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# More model
import xgboost as xgb
from skopt import BayesSearchCV
import statsmodels.api as sm

# %%
## Read dataframes ----
df_casos_uci = load('storage/df_export_cases_uci.joblib')
df_casos_uci_num_defunciones = load('storage/df_export_cases_uci_num_defunciones.joblib')
df_casos = load('storage/df_export_cases.joblib')
df_aemet = load('storage/df_export_aemet.joblib')
df_googletrends = load('storage/df_export_googletrends.joblib')
df_mitma = load('storage/df_export_mitma.joblib')
df_mscbs = load('storage/df_export_mscbs.joblib')
df_holidays = load('storage/df_export_holidays.joblib')

# %%
## Constants ----
TRAIN_PERCENTAGE = 0.9
MAX_LAG = 21
LAG_PANDEMIC = [1, 7, 14, MAX_LAG]
LAG_OTHER = [0, 7, 14, MAX_LAG]
LAG_PCT_CHANGE = [7, 14]
SEED = 42
PREDICTION_WINDOW = 7

# %%
## Prepare dataframes ----
### Target variable ----
df_casos_uci_num_defunciones = df_casos_uci_num_defunciones.add_prefix('uci_defun__')
### Death age groups ----
# Time shifting, minimum has to be `1`
# df_casos_uci_lagged = helpers.pct_change_by_lags(df_casos_uci, fix_columns=['fecha'], lag_numbers=LAG_PCT_CHANGE)
df_casos_uci_lagged = helpers.shift_timeseries_by_lags(df_casos_uci, fix_columns=['fecha'], lag_numbers=LAG_PANDEMIC)
df_casos_uci_lagged = df_casos_uci_lagged.add_prefix('uci__')
### Cases tested ----
# df_casos_lagged = helpers.pct_change_by_lags(df_casos, fix_columns=['fecha'], lag_numbers=LAG_PCT_CHANGE)
df_casos_lagged = helpers.shift_timeseries_by_lags(df_casos, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_casos_lagged = df_casos_lagged.add_prefix('tests__')
### AEMET temperature ----
# df_aemet_lagged = helpers.pct_change_by_lags(df_aemet, fix_columns=['fecha'], lag_numbers=LAG_PCT_CHANGE)
df_aemet_lagged = helpers.shift_timeseries_by_lags(df_aemet, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_aemet_lagged = df_aemet_lagged.add_prefix('aemet__')
### Google Trends ----
# df_googletrends_lagged = helpers.pct_change_by_lags(df_googletrends, fix_columns=['date'], lag_numbers=LAG_PCT_CHANGE)
df_googletrends_lagged = helpers.shift_timeseries_by_lags(df_googletrends, fix_columns=['date'], lag_numbers=LAG_OTHER)
df_googletrends_lagged = df_googletrends_lagged.add_prefix('google_trends__')
### Movements to Madrid (`mitma`) ----
# df_mitma_lagged = helpers.pct_change_by_lags(df_mitma, fix_columns=['fecha'], lag_numbers=LAG_PCT_CHANGE)
df_mitma_lagged = helpers.shift_timeseries_by_lags(df_mitma, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_mitma_lagged = df_mitma_lagged.add_prefix('mitma__')
### Vaccination in Spain (`mscbs`) ----
# df_mscbs_lagged = helpers.pct_change_by_lags(df_mscbs, fix_columns=['date'], lag_numbers=LAG_PCT_CHANGE)
df_mscbs_lagged = helpers.shift_timeseries_by_lags(df_mscbs, fix_columns=['date'], lag_numbers=LAG_OTHER)
df_mscbs_lagged = df_mscbs_lagged.add_prefix('mscbs__')
### Holidays ----
df_holidays_lagged = helpers.shift_timeseries_by_lags(df_holidays, fix_columns=['Date'], lag_numbers=LAG_OTHER)
df_holidays_lagged = df_holidays_lagged.add_prefix('holidays__')

# %%
## Merge all datasources ----
df_merge = df_casos_uci_num_defunciones.copy()
df_merge = df_merge.rename({'uci_defun__fecha': 'fecha'}, axis=1)

def merge_df_to_add(df_merge, df_to_add, date_column):
    df = df_merge.merge(df_to_add, left_on='fecha', right_on=date_column, how='inner')
    df = df.drop(columns=date_column)
    return df

df_merge = merge_df_to_add(df_merge, df_casos_uci_lagged, 'uci__fecha')
df_merge = merge_df_to_add(df_merge, df_casos_lagged, 'tests__fecha')
df_merge = merge_df_to_add(df_merge, df_aemet_lagged, 'aemet__fecha')
df_merge = merge_df_to_add(df_merge, df_googletrends_lagged, 'google_trends__date')
df_merge = merge_df_to_add(df_merge, df_mitma_lagged, 'mitma__fecha')
df_merge = merge_df_to_add(df_merge, df_mscbs_lagged, 'mscbs__date')
df_merge = merge_df_to_add(df_merge, df_holidays_lagged, 'holidays__Date')

# Replace `Inf`
# df_merge = df_merge.replace([np.inf, -np.inf], np.nan)

# Check for `nan` or `null`
# df_merge.isna().sum().sum()
# Fill `na` because of the `outer`
df_merge = df_merge.fillna(df_merge.mean(numeric_only=True))
# What cannot be filled by the `mean`, is filled by `0` 
# df_merge = df_merge.fillna(0)

# %%
## Feature engineering ----
### Filter only vaccination times
# idx_filter = df_merge['fecha'] >= helpers.start_date_vaccination
# idx_filter.value_counts()
# df_merge = df_merge.loc[idx_filter]

### Dates ----
# TODO: Fix `bisiesto`
df_merge['fecha_year'] = df_merge['fecha'].dt.year
df_merge['fecha_month'] = df_merge['fecha'].dt.month
df_merge['fecha_day'] = df_merge['fecha'].dt.day
df_merge['fecha_dayofweek'] = df_merge['fecha'].dt.dayofweek
df_merge['fecha_dayofyear'] = df_merge['fecha'].dt.dayofyear
df_merge['fecha_weekend'] = [(x in [5,6])*1 for x in df_merge['fecha_dayofweek']]
df_merge = df_merge.drop(columns=['fecha'])

# %%
## Split dataset ----
df_merge = df_merge.reset_index(drop=True)
# 90% for the training
# split_ = int(df_merge.shape[0]*TRAIN_PERCENTAGE)
split_ = df_merge.shape[0]-PREDICTION_WINDOW
# Split
df_train = df_merge.iloc[MAX_LAG:split_, :]
df_test = df_merge.iloc[split_:(split_+PREDICTION_WINDOW), :]

# X and y split
X_train = df_train.drop(columns='uci_defun__num_def')
y_train = df_train['uci_defun__num_def']

X_test = df_test.drop(columns='uci_defun__num_def')
y_test = df_test['uci_defun__num_def']

# %%
## Model training ----
### RandomForestRegressor ----
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=SEED, max_features='auto', n_estimators=100, n_jobs=-1)
rfr.fit(X_train, y_train)
y_train_pred = rfr.predict(X_train)
y_test_pred = rfr.predict(X_test)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
### RandomForestRegressor: pipeline ----
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

rfr_pipeline = make_pipeline(
    PCA(n_components=0.999999, svd_solver='full'), 
    StandardScaler(),
    RandomForestRegressor(random_state=SEED, max_features='auto', n_estimators=1000, n_jobs=-1))

rfr_pipeline.fit(X_train, y_train)
y_train_pred = rfr_pipeline.predict(X_train)
y_test_pred = rfr_pipeline.predict(X_test)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
### SVM ----
from sklearn.svm import SVR
regr = make_pipeline(StandardScaler(), SVR(kernel='linear'))
regr.fit(X_train, y_train)
y_train_pred = regr.predict(X_train)
y_test_pred = regr.predict(X_test)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
### LinearRegression ----
from sklearn.linear_model import LinearRegression
regr = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1))
regr.fit(X_train, y_train)
y_train_pred = regr.predict(X_train)
y_test_pred = regr.predict(X_test)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
## SequentialFeatureSelector ----
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

# Build the model
rfr = RandomForestRegressor(random_state=SEED, max_features=None, n_estimators=100, n_jobs=-1)
tss = TimeSeriesSplit(n_splits=3, test_size=PREDICTION_WINDOW)

# Documentation
# https://scikit-learn.org/stable/modules/model_evaluation.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# sfs = SequentialFeatureSelector(rfr, n_features_to_select=20, n_jobs=-1, scoring='r2', cv=tss)
sfs = RFECV(rfr, step=50, n_jobs=-1, scoring='r2', cv=tss)

start_time = helpers.timer(None)
sfs.fit(X_train, y_train)
helpers.timer(start_time)

# %%
# dump(sfs, 'storage/model.joblib')
# sfs = load('storage/model.joblib')

# %%
### Metric calculation ----
y_train_pred = sfs.predict(X_train)
y_test_pred = sfs.predict(X_test)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
# Number of features
sfs.n_features_

# %%
import matplotlib.pyplot as plt

ax = pd.DataFrame({'test': y_test.to_list(), 'pred': y_test_pred.tolist()}).plot(x='test', y='pred', kind='scatter')
# x = np.linspace(*ax.get_xlim())
# ax.plot(x, x)
ax.axline([0, 0], [1, 1])
plt.show()

# %%
## Feature importance ----
### Standard plot ----
# https://stackoverflow.com/a/51520906/3780957
feat_importances = pd.Series(sfs.estimator_.feature_importances_, index=X_train.columns[sfs.support_])
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()

# %%
### Shap ----
import shap
explainer = shap.TreeExplainer(sfs.estimator_)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# %%
X_test_shap = X_test.loc[:, sfs.support_]
explainer = shap.KernelExplainer(sfs.estimator_.predict, X_test_shap)
shap_values = explainer.shap_values(X_test_shap, approximate=False, check_additivity=False)
shap.summary_plot(shap_values, X_test_shap)

# %%
# TODO: Correlation, https://gist.github.com/aigera2007/567a6d34cefb30c7c6255c20e40f24fb/raw/9c9cb058d1e00533b7dd9dc8f0fd9d3ad03caabb/corr_matrix.py
# TODO: LSTM
# TODO: SHAP 
# https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
# https://aigerimshopenova.medium.com/random-forest-classifier-and-shap-how-to-understand-your-customers-and-interpret-a-black-box-model-6166d86820d9