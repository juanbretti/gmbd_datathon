# -*- coding: utf-8 -*-

# Analytical Design (10%)
# TODO: Coherence use case	0,05
# TODO: Business needs	0,05
# Data understanding and data processing (10%)
# TODO: Data processing pipeline diagram	0,05
# TODO: More than one external data source	0,05
# Analysis and models (60%)
# TODO: Exploratory data analysis	0,1
# TODO: Use of whole period	0,1
# TODO: Expl. model R2	0,1
# TODO: Beta coeff. viz or table (comparison between variable groups)	0,2
# TODO: Predictive model R2	0,1
# Presentation & Report (20%)
# TODO: Present. Quality	0,1
# TODO: Report Quality	0,1

# %%
## Libraries ----
# General usage
from os import error
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
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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
TARGET_VARIABLE = 'casos_uci_target__ratio_population__num_def'
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
df_casos_uci_target = add_prefix(df_casos_uci_target, 'casos_uci_target__')
### Death age groups ----
df_casos_uci_lagged = helpers.shift_timeseries_by_lags(df_casos_uci, fix_columns=fix_columns, lag_numbers=LAG_UCI)
df_casos_uci_lagged = add_prefix(df_casos_uci_lagged, 'casos_uci__')
### Cases tested ----
df_casos_lagged = helpers.shift_timeseries_by_lags(df_casos, fix_columns=fix_columns, lag_numbers=LAG_CASOS)
df_casos_lagged = add_prefix(df_casos_lagged, 'casos__')
### AEMET temperature ----
df_aemet_lagged = helpers.shift_timeseries_by_lags(df_aemet, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_aemet_lagged = add_prefix(df_aemet_lagged, 'aemet__')
### Google Trends ----
df_googletrends_lagged = helpers.shift_timeseries_by_lags(df_googletrends, fix_columns=fix_columns, lag_numbers=LAG_OTHER)
df_googletrends_lagged = add_prefix(df_googletrends_lagged, 'googletrends__')
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
## All possible column combinations ----
import itertools

sources = ['casos_uci', 'casos', 'aemet', 'googletrends', 'mitma', 'mscbs', 'holidays']
all_combinations = []
for i in range(1, len(sources)+1):
    all_combinations = all_combinations + list(itertools.combinations(sources, i))

## Calculate model for combination ----
def model_for_combination(combination, consider_province=False):

    # Merge the datasources
    df_merge = df_casos_uci_target.copy()
    if 'casos_uci' in combination:
        df_merge = df_merge.merge(df_casos_uci_lagged, on=fix_columns)
    if 'casos' in combination:
        df_merge = df_merge.merge(df_casos_lagged, on=fix_columns)
    if 'aemet' in combination:
        df_merge = df_merge.merge(df_aemet_lagged, on=fix_columns)
    if 'googletrends' in combination:
        df_merge = df_merge.merge(df_googletrends_lagged, on=fix_columns)
    if 'mitma' in combination:
        df_merge = df_merge.merge(df_mitma_lagged, on=fix_columns)
    if 'mscbs' in combination:
        df_merge = df_merge.merge(df_mscbs_lagged, on=fix_columns)
    if 'holidays' in combination:
        df_merge = df_merge.merge(df_holidays_lagged, on=fix_columns)

    # Feature engineering of `fecha`
    df_merge['fecha_year'] = df_merge['fecha'].dt.year
    df_merge['fecha_month'] = df_merge['fecha'].dt.month
    df_merge['fecha_day'] = df_merge['fecha'].dt.day
    df_merge['fecha_dayofweek'] = df_merge['fecha'].dt.dayofweek
    df_merge['fecha_dayofyear'] = df_merge['fecha'].dt.dayofyear
    df_merge['fecha_weekend'] = [(x in [5,6])*1 for x in df_merge['fecha_dayofweek']]

    # Remove location information
    if not consider_province:
        df_merge = df_merge.drop(columns=['Code provincia alpha', 'Code comunidad autónoma alpha'], errors='ignore').groupby('fecha').sum()
        df_merge = df_merge.reset_index()

    # Train test split
    df_merge = df_merge.dropna().reset_index(drop=True)
    ## Last `PREDICTION_WINDOW` days
    # df_test = df_merge.sort_values(['fecha']).groupby(['Code provincia alpha']).tail(PREDICTION_WINDOW)
    # df_train = df_merge[~df_merge.index.isin(df_test.index)]
    ## Random split
    if consider_province:
        prediction_window_ = PREDICTION_WINDOW*df_merge['Code provincia alpha'].nunique()
        df_train, df_test = train_test_split(df_merge, test_size=prediction_window_, random_state=42, stratify=df_merge['Code provincia alpha'])
    else:
        df_train, df_test = train_test_split(df_merge, test_size=PREDICTION_WINDOW, random_state=42)

    X_train = df_train.drop(columns=TARGET_VARIABLE)
    X_test = df_test.drop(columns=TARGET_VARIABLE)
    y_train = df_train[TARGET_VARIABLE].reset_index(drop=True)
    y_test = df_test[TARGET_VARIABLE].reset_index(drop=True)

    # Dummies
    if consider_province:
        X_train_dummies = pd.get_dummies(X_train, columns = ['Code provincia alpha'], drop_first=True)
    else:
        X_train_dummies = X_train
    X_train_dummies = X_train_dummies.drop(columns=fix_columns, errors='ignore')
    # Scaler
    scaler = StandardScaler()
    X_tranformed = scaler.fit_transform(X_train_dummies)
    X_train_dummies_scaled = pd.DataFrame(X_tranformed, columns=X_train_dummies.columns)

    # Dummies
    if consider_province:
        X_test_dummies = pd.get_dummies(X_test, columns = ['Code provincia alpha'], drop_first=True)
        X_test_dummies = X_test_dummies.reindex(columns = X_train_dummies.columns, fill_value=0)  # Ensure that I have all the columns in the `dummy`.
    else:
        X_test_dummies = X_test
    X_test_dummies = X_test_dummies.drop(columns=fix_columns, errors='ignore')
    # Scaler
    X_tranformed = scaler.transform(X_test_dummies)
    X_test_dummies_scaled = pd.DataFrame(X_tranformed, columns=X_test_dummies.columns)

    # Regression model
    X_train_dummies_scaled = sm.add_constant(X_train_dummies_scaled)
    X_test_dummies_scaled = sm.add_constant(X_test_dummies_scaled, has_constant='add')
    model = sm.OLS(y_train, X_train_dummies_scaled)
    results = model.fit()

    # Predict
    y_train_pred = results.predict(X_train_dummies_scaled)
    y_test_pred = results.predict(X_test_dummies_scaled)

    # Append results
    # results.rsquared_adj
    df_combination = pd.DataFrame({
        'Combination': [str(combination)], 
        'Combination length': len(combination), 
        'Number of features': X_train.shape[1], 
        'R2 train': metrics.r2_score(y_train, y_train_pred), 
        'R2 test': metrics.r2_score(y_test, y_test_pred)})

    df_coefficients = pd.DataFrame({
        'Combination': str(combination), 
        'Feature': results.pvalues.keys(),
        'Coefficient': results.params.values, 
        'p-value': results.pvalues.values})
        
    return df_combination, df_coefficients, combination, model, results, X_train_dummies_scaled, y_train, X_test_dummies_scaled, y_test

# %%
### Run ----
df_all_combinations, df_all_coefficients = pd.DataFrame(), pd.DataFrame()

for combination in all_combinations:
    df_combination, df_coefficients = model_for_combination(combination, False)[0:2]
    df_all_combinations = df_all_combinations.append(df_combination, ignore_index=True)
    df_all_coefficients = df_all_coefficients.append(df_coefficients, ignore_index=True)

# %%
## Coefficients table ----
# TODO: Beta coeff. viz or table (comparison between variable groups)	0,2
df_all_coefficients_pivot = pd.pivot(df_all_coefficients, index=['Combination'], columns=['Feature'], values=['Coefficient'])
df_all_coefficients_pivot

# %%
### Sort values ----
pd.set_option('display.max_rows', 100)
df_all_combinations = df_all_combinations.sort_values(by=['R2 test', 'Combination length'], ascending=[False, True])
df_all_combinations

t1 = df_all_combinations[df_all_combinations['Combination length'] == 1]
t2 = df_all_combinations[(df_all_combinations['Combination length'] == 2) & (['casos_uci' in x for x in df_all_combinations['Combination']])]
t3 = df_all_combinations[(df_all_combinations['Combination length'] == 3) & (['casos_uci' in x for x in df_all_combinations['Combination']])]

# %%
### Coefficients per case ----
df_all_coefficients_pivot[df_all_coefficients_pivot.index.isin(t2['Combination'])].dropna(how='all').T

# %%
### Testing ----
_, _, _, _, _, X_train_dummies_scaled, y_train, X_test_dummies_scaled, y_test = model_for_combination(('mscbs', 'holidays'), False)

# %%
# https://planspace.org/20150423-forward_selection_with_statsmodels/

# %%
## SequentialFeatureSelector ----
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

# Build the model
rfr = RandomForestRegressor(random_state=SEED, max_features=None, n_estimators=100, n_jobs=-1)
fs = RFECV(rfr, step=50, n_jobs=-1, scoring='r2', cv=4)

start_time = helpers.timer(None)
fs.fit(X_train_dummies_scaled, y_train)
helpers.timer(start_time)

# %%
### Metric calculation ----
y_train_pred = fs.predict(X_train_dummies_scaled)
y_test_pred = fs.predict(X_test_dummies_scaled)
helpers.metrics_custom2(y_train, y_train_pred, y_test, y_test_pred)

# %%
# Number of features
fs.n_features_

# %%
ax = pd.DataFrame({'test': y_test.to_list(), 'pred': y_test_pred.tolist()}).plot(x='test', y='pred', kind='scatter')
ax.axline([0, 0], [1, 1])
plt.show()

# %%
## Feature importance ----
### Standard plot ----
# https://stackoverflow.com/a/51520906/3780957
feat_importances = pd.Series(fs.estimator_.feature_importances_, index=X_train_dummies_scaled.columns[fs.support_])
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()

# %%
### Shap ----
import shap
explainer = shap.TreeExplainer(fs.estimator_)
shap_values = explainer.shap_values(X_train_dummies_scaled)
shap.summary_plot(shap_values, X_train_dummies_scaled, plot_type="bar")

# %%
X_test_shap = X_train_dummies_scaled.loc[:, fs.support_]
explainer = shap.KernelExplainer(fs.estimator_.predict, X_test_shap)
shap_values = explainer.shap_values(X_test_shap, approximate=False, check_additivity=False)
shap.summary_plot(shap_values, X_test_shap)

# %%