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
import itertools

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
## Constants ----
TARGET_VARIABLE_1 = 'ratio_population__num_def__lag_-7'
TARGET_VARIABLE_2 = f'casos_uci_target__{TARGET_VARIABLE_1}'
LAG_TARGET = [-7]
LAG_UCI = [0, 7, 14, 21]
LAG_CASOS = [0, 7, 14, 21]
LAG_OTHER = [0, 7, 14, 21]
LAG_MAX = max(LAG_UCI+LAG_CASOS+LAG_OTHER)
SEED = 42
PREDICTION_WINDOW = 7
PROPORTION_TEST = 0.3
FIX_COLUMNS = ['fecha', 'Code provincia alpha', 'Code comunidad autónoma alpha']
P_VALUE = 0.05

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
## Prepare dataframes ----
### Target variable ----
df_casos_uci_lagged = helpers.shift_timeseries_by_lags(df_casos_uci, FIX_COLUMNS, LAG_TARGET)
df_casos_uci_target = df_casos_uci_lagged[FIX_COLUMNS+[TARGET_VARIABLE_1]]
df_casos_uci_target = helpers.add_prefix(df_casos_uci_target, 'casos_uci_target__', FIX_COLUMNS)
### Death age groups ----
df_casos_uci = df_casos_uci.drop(columns=['ratio_population__num_def'])
df_casos_uci_lagged = helpers.shift_timeseries_by_lags(df_casos_uci, FIX_COLUMNS, LAG_UCI)
df_casos_uci_lagged = helpers.add_prefix(df_casos_uci_lagged, 'casos_uci__', FIX_COLUMNS)
### Cases tested ----
df_casos_lagged = helpers.shift_timeseries_by_lags(df_casos, FIX_COLUMNS, LAG_CASOS)
df_casos_lagged = helpers.add_prefix(df_casos_lagged, 'casos__', FIX_COLUMNS)
### AEMET temperature ----
df_aemet_lagged = helpers.shift_timeseries_by_lags(df_aemet, FIX_COLUMNS, LAG_OTHER)
df_aemet_lagged = helpers.add_prefix(df_aemet_lagged, 'aemet__', FIX_COLUMNS)
### Google Trends ----
df_googletrends_lagged = helpers.shift_timeseries_by_lags(df_googletrends, FIX_COLUMNS, LAG_OTHER)
df_googletrends_lagged = helpers.add_prefix(df_googletrends_lagged, 'googletrends__', FIX_COLUMNS)
### Movements to Madrid (`mitma`) ----
df_mitma_lagged = helpers.shift_timeseries_by_lags(df_mitma, FIX_COLUMNS, LAG_OTHER)
df_mitma_lagged = helpers.add_prefix(df_mitma_lagged, 'mitma__', FIX_COLUMNS)
### Vaccination in Spain (`mscbs`) ----
df_mscbs_lagged = helpers.shift_timeseries_by_lags(df_mscbs, FIX_COLUMNS, LAG_OTHER)
df_mscbs_lagged = helpers.add_prefix(df_mscbs_lagged, 'mscbs__', FIX_COLUMNS)
### Holidays ----
df_holidays_lagged = helpers.shift_timeseries_by_lags(df_holidays, FIX_COLUMNS, LAG_OTHER)
df_holidays_lagged = helpers.add_prefix(df_holidays_lagged, 'holidays__', FIX_COLUMNS)

# %%
## All possible column combinations ----
# 'casos_uci', 'casos', 
sources = ['aemet', 'googletrends', 'mitma', 'mscbs', 'holidays']
all_combinations = []
for i in range(1, len(sources)+1):
    all_combinations = all_combinations + list(itertools.combinations(sources, i))

combination = ('casos_uci', 'casos', 'googletrends')
p_value=0.05

## Calculate model for combination ----
def model_for_combination(combination):

    # Merge the datasources
    df_merge = df_casos_uci_target.copy()
    if 'casos_uci' in combination:
        df_merge = df_merge.merge(df_casos_uci_lagged, on=FIX_COLUMNS)
    if 'casos' in combination:
        df_merge = df_merge.merge(df_casos_lagged, on=FIX_COLUMNS)
    if 'aemet' in combination:
        df_merge = df_merge.merge(df_aemet_lagged, on=FIX_COLUMNS)
    if 'googletrends' in combination:
        df_merge = df_merge.merge(df_googletrends_lagged, on=FIX_COLUMNS)
    if 'mitma' in combination:
        df_merge = df_merge.merge(df_mitma_lagged, on=FIX_COLUMNS)
    if 'mscbs' in combination:
        df_merge = df_merge.merge(df_mscbs_lagged, on=FIX_COLUMNS)
    if 'holidays' in combination:
        df_merge = df_merge.merge(df_holidays_lagged, on=FIX_COLUMNS)

    # Remove location information
    df_merge = df_merge.drop(columns=['Code provincia alpha', 'Code comunidad autónoma alpha'], errors='ignore').groupby('fecha').sum()
    df_merge = df_merge.drop(columns=FIX_COLUMNS, errors='ignore')
    df_merge = df_merge.dropna().reset_index(drop=True)

    # Train test split
    df_train, df_test = train_test_split(df_merge, test_size=PROPORTION_TEST, random_state=42)
    X_train = df_train.drop(columns=TARGET_VARIABLE_2)
    X_test = df_test.drop(columns=TARGET_VARIABLE_2)
    y_train = df_train[TARGET_VARIABLE_2].reset_index(drop=True)
    y_test = df_test[TARGET_VARIABLE_2].reset_index(drop=True)

    # Scaler
    scaler = StandardScaler()
    X_tranformed = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_tranformed, columns=X_train.columns)
    X_tranformed = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_tranformed, columns=X_test.columns)

    # Regression model
    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled, has_constant='add')
    model = sm.OLS(y_train, X_train_scaled)
    results = model.fit()

    # Filter by `p-value`
    results_filtered = results.pvalues[results.pvalues.values<=P_VALUE]
    results_filtered_col = results_filtered.keys()

    # New model, after filter
    model = sm.OLS(y_train, X_train_scaled[results_filtered_col])
    results = model.fit()

    # Predict
    y_train_pred = results.predict(X_train_scaled[results_filtered_col])
    y_test_pred = results.predict(X_test_scaled[results_filtered_col])

    # Append results
    # results.rsquared_adj
    df_combination = pd.DataFrame({
        'Combination': str(combination), 
        'Combination length': len(combination), 
        'Number of features': len(results_filtered_col), #X_train.shape[1], 
        'R2 train': metrics.r2_score(y_train, y_train_pred), 
        'R2 test': metrics.r2_score(y_test, y_test_pred)}, index=[0])

    df_coefficients = pd.DataFrame({
        'Combination': str(combination), 
        'Feature': results.pvalues.keys(),
        'Coefficient': results.params.values, 
        'p-value': results.pvalues.values})
        
    return df_combination, df_coefficients, combination, model, results, X_train_scaled, y_train, X_test_scaled, y_test

# %%
### Run ----
df_all_combinations, df_all_coefficients = pd.DataFrame(), pd.DataFrame()

for combination in all_combinations:
    print(f'Calculating: {str(combination)}')
    df_combination, df_coefficients = model_for_combination(combination)[0:2]
    df_all_combinations = df_all_combinations.append(df_combination, ignore_index=True)
    df_all_coefficients = df_all_coefficients.append(df_coefficients, ignore_index=True)

# %%
## Coefficients table ----
# TODO: Beta coeff. viz or table (comparison between variable groups)	0,2
df_all_coefficients_pivot = pd.pivot(df_all_coefficients, index=['Combination'], columns=['Feature'], values=['Coefficient'])
# df_all_coefficients_pivot

# %%
### Sort values ----
# pd.set_option('display.max_rows', 100)
df_all_combinations = df_all_combinations.sort_values(by=['R2 test', 'Combination length'], ascending=[False, True])
df_all_combinations.head(20)

# %%
t1 = df_all_combinations[df_all_combinations['Combination length'] == 1]
t2 = df_all_combinations[(df_all_combinations['Combination length'] == 2) & (['googletrends' in x for x in df_all_combinations['Combination']])]
t3 = df_all_combinations[(df_all_combinations['Combination length'] == 3) & (['googletrends' in x for x in df_all_combinations['Combination']])]

# %%
### Coefficients per case ----
# df_all_coefficients_pivot[df_all_coefficients_pivot.index.isin(t1['Combination'])].dropna(how='all').T

# %%
### Testing ----
df_combination, df_coefficients, _, _, _, X_train_dummies_scaled, y_train, X_test_dummies_scaled, y_test = model_for_combination(('casos_uci', 'casos', 'googletrends'), False)
df_coefficients = df_coefficients.sort_values('Coefficient', ascending=True)

df_coefficients.plot.barh(x='Feature', y='Coefficient', figsize=[10,20]).invert_yaxis()
# df_coefficients_filtered = df_coefficients[df_coefficients['p-value']<=0.05]
# df_coefficients_filtered.plot.barh(x='Feature', y='Coefficient', figsize=[10,20]).invert_yaxis()

# _, _, _, _, _, X_train_dummies_scaled, y_train, X_test_dummies_scaled, y_test = model_for_combination(('mscbs', 'holidays'), False)

# %%
# Filter
df_coefficients_filtered = df_coefficients[df_coefficients['p-value']<=0.05]

# Data filtered
X_train = X_train_dummies_scaled[df_coefficients_filtered['Feature']]
X_test = X_test_dummies_scaled[df_coefficients_filtered['Feature']]

# Regression model
model = sm.OLS(y_train, X_train)
results = model.fit()

# Predict
y_train_pred = results.predict(X_train)
y_test_pred = results.predict(X_test)

# Append results
# results.rsquared_adj
df_combination = pd.DataFrame({
    'Number of features': [X_train.shape[1]], 
    'R2 train': [metrics.r2_score(y_train, y_train_pred)], 
    'R2 test': [metrics.r2_score(y_test, y_test_pred)]})

df_coefficients = pd.DataFrame({
    'Feature': results.pvalues.keys(),
    'Coefficient': results.params.values, 
    'p-value': results.pvalues.values})

# %%
df_coefficients = df_coefficients.sort_values('Coefficient', ascending=True)
df_coefficients.plot.barh(x='Feature', y='Coefficient', figsize=[10,20]).invert_yaxis()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

sns.set(style='white', font_scale=1.6)
g = sns.PairGrid(X_train.iloc[:, 0:5], aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)

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