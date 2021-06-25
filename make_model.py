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
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

# Model
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Linear model
import statsmodels.api as sm


# %%
## Read dataframes ----
df_casos_uci = load('storage/df_export_casos_uci.joblib')
df_casos = load('storage/df_export_casos.joblib')
df_aemet = load('storage/df_export_aemet.joblib')
df_googletrends = load('storage/df_export_googletrends.joblib')
df_mitma = load('storage/df_export_mitma.joblib')

### Province ----
province_code = helpers.province_code()
province_code = province_code[['Code provincia alpha', 'Code comunidad autónoma alpha']].drop_duplicates()

# %%
## Constants ----
TARGET_PROVINCE = 'M'
TRAIN_PERCENTAGE = 0.9
TUNNING_PARAM_COMB = 10
MAX_LAG = 21
LAG_PANDEMIC = [1, 7, 14, MAX_LAG]
LAG_OTHER = [0, 7, 14, MAX_LAG]
SEED = 42

# %%
## Prepare dataframes ----
### Target variable ----
filter_ = df_casos_uci['provincia_iso']==TARGET_PROVINCE
df_casos_uci_num_defunciones = df_casos_uci[filter_].groupby(['fecha']).aggregate({'num_def': np.sum})

df_casos_uci_num_defunciones = df_casos_uci_num_defunciones.reset_index()
df_casos_uci_num_defunciones = df_casos_uci_num_defunciones.add_prefix('uci_defun__')

# %%
### Death age groups ----
# Merge groups
df_casos_uci['grupo_edad_merged'] = df_casos_uci['grupo_edad'].replace({'0-9': '0-59', '10-19': '0-59', '20-29': '0-59', '30-39': '0-59', '40-49': '0-59', '50-59': '0-59', 'NC': '0-59'})
# Testing groups sizes
df_casos_uci.groupby('grupo_edad_merged').sum()
# Merge CA information
df_casos_uci = df_casos_uci.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
# Pivot per CA
df_casos_uci_age = pd.pivot_table(df_casos_uci, index=['fecha'], columns=['Code comunidad autónoma alpha', 'grupo_edad_merged'], values=['num_hosp', 'num_uci', 'num_def'], aggfunc=np.sum, fill_value=0)
# Flatten column names and remove index
df_casos_uci_age.columns = ['__'.join(x) for x in df_casos_uci_age.columns]
df_casos_uci_age = df_casos_uci_age.reset_index()
# Time shifting, minimum has to be `1`
df_casos_uci_age_lagged = helpers.shift_timeseries_by_lags(df_casos_uci_age, fix_columns=['fecha'], lag_numbers=LAG_PANDEMIC)
df_casos_uci_age_lagged = df_casos_uci_age_lagged.add_prefix('uci_age__')

# %%
### Cases tested ----
df_casos_pruebas = df_casos.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
df_casos_pruebas = pd.pivot_table(df_casos_pruebas, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['num_casos_prueba_pcr', 'num_casos_prueba_test_ac', 'num_casos_prueba_ag', 'num_casos_prueba_elisa', 'num_casos_prueba_desconocida'], aggfunc=np.sum, fill_value=0)

df_casos_pruebas.columns = ['__'.join(x) for x in df_casos_pruebas.columns]
df_casos_pruebas = df_casos_pruebas.reset_index()
df_casos_pruebas_lagged = helpers.shift_timeseries_by_lags(df_casos_pruebas, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_casos_pruebas_lagged = df_casos_pruebas_lagged.add_prefix('tests__')

# %%
### AEMET temperature ----
df_aemet_pivot = df_aemet[[('tmed', 'mean'), ]]
df_aemet_pivot = df_aemet_pivot.reset_index()
df_aemet_pivot.columns = [x[0] for x in df_aemet_pivot.columns]
df_aemet_pivot = df_aemet_pivot.merge(province_code, on='Code provincia alpha')
df_aemet_pivot = pd.pivot_table(df_aemet_pivot, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['tmed'], aggfunc=np.sum, fill_value=0)

df_aemet_pivot.columns = ['__'.join(x) for x in df_aemet_pivot.columns]
df_aemet_pivot = df_aemet_pivot.reset_index()
df_aemet_pivot_lagged = helpers.shift_timeseries_by_lags(df_aemet_pivot, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_aemet_pivot_lagged = df_aemet_pivot_lagged.add_prefix('aemet__')

# %%
### Google Trends ----
df_googletrends_pivot = df_googletrends
df_googletrends_pivot = df_googletrends_pivot.pivot(index=['date'], columns=['ca'])

df_googletrends_pivot.columns = ['__'.join(x) for x in df_googletrends_pivot.columns]
df_googletrends_pivot = df_googletrends_pivot.reset_index()
df_googletrends_pivot_lagged = helpers.shift_timeseries_by_lags(df_googletrends_pivot, fix_columns=['date'], lag_numbers=LAG_OTHER)
df_googletrends_pivot_lagged = df_googletrends_pivot_lagged.add_prefix('google_trends__')

# %%
### Movements to Madrid (`mitma`) ----
filter_ = df_mitma['destino_province']==TARGET_PROVINCE
df_mitma_filtered = df_mitma[filter_]
df_mitma_filtered = df_mitma_filtered.drop(columns='destino_province')
df_mitma_filtered_lagged = helpers.shift_timeseries_by_lags(df_mitma_filtered, fix_columns=['fecha'], lag_numbers=LAG_OTHER)
df_mitma_filtered_lagged = df_mitma_filtered_lagged.add_prefix('mitma__')

# %%
## Merge all datasources ----
df_merge = df_casos_uci_num_defunciones.copy()
df_merge = df_merge.rename({'uci_defun__fecha': 'fecha'}, axis=1)

def merge_df_to_add(df_merge, df_to_add, date_column):
    df = df_merge.merge(df_to_add, left_on='fecha', right_on=date_column, how='inner')
    df = df.drop(columns=date_column)
    return df

df_merge = merge_df_to_add(df_merge, df_casos_uci_age_lagged, 'uci_age__fecha')
df_merge = merge_df_to_add(df_merge, df_casos_pruebas_lagged, 'tests__fecha')
df_merge = merge_df_to_add(df_merge, df_aemet_pivot_lagged, 'aemet__fecha')
df_merge = merge_df_to_add(df_merge, df_googletrends_pivot_lagged, 'google_trends__date')
df_merge = merge_df_to_add(df_merge, df_mitma_filtered_lagged, 'mitma__fecha')

# Fill `na` because of the `outer`
df_merge = df_merge.fillna(df_merge.mean())
# Check for `nan` or `null`
# df_merge.isnull().sum().values.sum()

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
split_ = int(df_merge.shape[0]*TRAIN_PERCENTAGE)
# Split
df_train = df_merge.iloc[MAX_LAG:split_, :]
df_test = df_merge.iloc[split_:(split_+7), :]

# X and y split
X_train = df_train.drop(columns='uci_defun__num_def')
y_train = df_train['uci_defun__num_def']

X_test = df_test.drop(columns='uci_defun__num_def')
y_test = df_test['uci_defun__num_def']

# %%
## Model training ----
### Quick model ----
rfr = RandomForestRegressor(random_state=SEED, max_features='auto', n_estimators=1000, n_jobs=-1)
rfr.fit(X_train, y_train)
y_train_pred = rfr.predict(X_train)
y_test_pred = rfr.predict(X_test)

# %%
### Metrics ----
print('** Train **')
print(f'Parson R2: {metrics.r2_score(y_train, y_train_pred)}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_train, y_train_pred)}')
print(f'Mean Absolute Percengage Error: {metrics.mean_absolute_percentage_error(y_train, y_train_pred)}')

print('** Test **')
print(f'Parson R2: {metrics.r2_score(y_test, y_test_pred)}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_test, y_test_pred)}')
print(f'Mean Absolute Percengage Error: {metrics.mean_absolute_percentage_error(y_test, y_test_pred)}')

# %%
## Feature importance ----
### Standard plot ----
# https://stackoverflow.com/a/51520906/3780957
feat_importances = pd.Series(rfr.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()

# %%
### Shap ----
import shap
explainer = shap.TreeExplainer(rfr)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# %%

# %%
# TODO: lag 7, 14 para todas
# TODO: Variables de pandemia, tienen que tener lag
# TODO: falta el censo
# TODO: faltan las vacunas
# TODO: MOdelo lineal
# TODO: Correlation
# https://gist.github.com/aigera2007/567a6d34cefb30c7c6255c20e40f24fb/raw/9c9cb058d1e00533b7dd9dc8f0fd9d3ad03caabb/corr_matrix.py
# TODO: SHAP 
# https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
# https://aigerimshopenova.medium.com/random-forest-classifier-and-shap-how-to-understand-your-customers-and-interpret-a-black-box-model-6166d86820d9
# TODO: hacer un standardscaler
