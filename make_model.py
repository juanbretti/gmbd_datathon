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
df_casos_uci = load('storage/df_export_casos_uci.joblib')
df_casos = load('storage/df_export_casos.joblib')
df_aemet = load('storage/df_export_aemet.joblib')
df_googletrends = load('storage/df_export_googletrends.joblib')
df_mitma = load('storage/df_export_mitma.joblib')
df_mscbs = load('storage/df_export_mscbs.joblib')

# %%
## Constants ----
TRAIN_PERCENTAGE = 0.9
MAX_LAG = 21
LAG_PANDEMIC = [1, 7, 14, MAX_LAG]
LAG_OTHER = [0, 7, 14, MAX_LAG]
SEED = 42
PREDICTION_WINDOW = 7

# %%
## Prepare dataframes ----
### Target variable ----
filter_ = df_casos_uci['provincia_iso']==helpers.target_province
df_casos_uci_num_defunciones = df_casos_uci[filter_].groupby(['fecha']).aggregate({'num_def': np.sum})
df_casos_uci_num_defunciones = df_casos_uci_num_defunciones.reset_index()
df_casos_uci_num_defunciones = df_casos_uci_num_defunciones.add_prefix('uci_defun__')

# %%
### Death age groups ----
# Merge groups
df_casos_uci['grupo_edad_merged'] = df_casos_uci['grupo_edad'].replace({'0-9': '0-59', '10-19': '0-59', '20-29': '0-59', '30-39': '0-59', '40-49': '0-59', '50-59': '0-59'})
# Testing groups sizes
# df_casos_uci.groupby('grupo_edad_merged').sum()
# Merge CA information
province_code = helpers.province_code()[['Code provincia alpha', 'Code comunidad autónoma alpha']].drop_duplicates()
df_casos_uci = df_casos_uci.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
# Pivot per CA
df_casos_uci_age = pd.pivot_table(df_casos_uci, index=['fecha'], columns=['Code comunidad autónoma alpha', 'grupo_edad_merged'], values=['num_hosp', 'num_uci', 'num_def'], aggfunc=np.sum, fill_value=0)
# Complete all the series
df_casos_uci_age = df_casos_uci_age.resample('d').ffill()
# Flatten column names and remove index
df_casos_uci_age.columns = ['__'.join(x) for x in df_casos_uci_age.columns]
df_casos_uci_age = df_casos_uci_age.reset_index()
# Time shifting, minimum has to be `1`
df_casos_uci_age_lagged = helpers.shift_timeseries_by_lags(df_casos_uci_age, fix_columns=['fecha'], lag_numbers=LAG_PANDEMIC)
df_casos_uci_age_lagged = df_casos_uci_age_lagged.add_prefix('uci_age__')

# %%
### Cases tested ----
df_casos_lagged = helpers.shift_timeseries_by_lags(df_casos, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_casos_lagged = df_casos_lagged.add_prefix('tests__')
### AEMET temperature ----
df_aemet_lagged = helpers.shift_timeseries_by_lags(df_aemet, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_aemet_lagged = df_aemet_lagged.add_prefix('aemet__')
### Google Trends ----
df_googletrends_lagged = helpers.shift_timeseries_by_lags(df_googletrends, fix_columns=['date'], lag_numbers=LAG_OTHER)
df_googletrends_lagged = df_googletrends_lagged.add_prefix('google_trends__')
### Movements to Madrid (`mitma`) ----
df_mitma_lagged = helpers.shift_timeseries_by_lags(df_mitma, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_mitma_lagged = df_mitma_lagged.add_prefix('mitma__')
### Vaccination in Spain (`mscbs`) ----
df_mscbs_lagged = helpers.shift_timeseries_by_lags(df_mscbs, fix_columns=['date'], lag_numbers=LAG_OTHER)
df_mscbs_lagged = df_mscbs_lagged.add_prefix('mscbs__')

# %%
## Merge all datasources ----
df_merge = df_casos_uci_num_defunciones.copy()
df_merge = df_merge.rename({'uci_defun__fecha': 'fecha'}, axis=1)

def merge_df_to_add(df_merge, df_to_add, date_column):
    df = df_merge.merge(df_to_add, left_on='fecha', right_on=date_column, how='inner')
    df = df.drop(columns=date_column)
    return df

df_merge = merge_df_to_add(df_merge, df_casos_uci_age_lagged, 'uci_age__fecha')
df_merge = merge_df_to_add(df_merge, df_casos_lagged, 'tests__fecha')
df_merge = merge_df_to_add(df_merge, df_aemet_lagged, 'aemet__fecha')
df_merge = merge_df_to_add(df_merge, df_googletrends_lagged, 'google_trends__date')
df_merge = merge_df_to_add(df_merge, df_mitma_lagged, 'mitma__fecha')
df_merge = merge_df_to_add(df_merge, df_mscbs_lagged, 'mscbs__date')

# Check for `nan` or `null`
# df_merge.isnull().sum().values.sum()
# Fill `na` because of the `outer`
df_merge = df_merge.fillna(df_merge.mean(numeric_only=True))

# %%
## Feature engineering ----
### Dates ----
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
rfr = RandomForestRegressor(random_state=SEED, max_features='auto', n_estimators=1000, n_jobs=-1)
rfr.fit(X_train, y_train)
y_train_pred = rfr.predict(X_train)
y_test_pred = rfr.predict(X_test)
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
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV

# Build the model
rfr = RandomForestRegressor(random_state=SEED, max_features=None, n_estimators=100, n_jobs=-1)
tss = TimeSeriesSplit(n_splits=3, test_size=PREDICTION_WINDOW)

# Documentation
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
sfs = load('storage/model.joblib')

# %%
### Metric calculation ----
y_train_pred = sfs.predict(X_train)
y_test_pred = sfs.predict(X_test)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
# Number of features
sfs.n_features_

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
# TODO: Mayor lag
# TODO: falta el censo
# TODO: Correlation, https://gist.github.com/aigera2007/567a6d34cefb30c7c6255c20e40f24fb/raw/9c9cb058d1e00533b7dd9dc8f0fd9d3ad03caabb/corr_matrix.py
# TODO: LTSM
# TODO: SHAP 
# https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
# https://aigerimshopenova.medium.com/random-forest-classifier-and-shap-how-to-understand-your-customers-and-interpret-a-black-box-model-6166d86820d9