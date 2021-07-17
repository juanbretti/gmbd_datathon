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
from pandas_profiling import ProfileReport

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
LAG_TARGET = [-7]
LAG_UCI = [1, 7, 14, 21]
LAG_CASOS = [0, 7, 14, 21]
LAG_OTHER = [10, 20, 30]
TARGET_VARIABLE_0 = 'ratio_population__num_casos'
TARGET_VARIABLE_1 = f'{TARGET_VARIABLE_0}__lag_{LAG_TARGET[0]}'
TARGET_VARIABLE_2 = f'casos_uci_target__{TARGET_VARIABLE_1}'
SEED = 42
PROPORTION_TEST = 0.3
FIX_COLUMNS = ['fecha', 'Code provincia alpha', 'Code comunidad autónoma alpha']
SOURCES = ['casos_uci', 'casos', 'aemet', 'googletrends', 'mitma', 'mscbs', 'holidays']
MANUAL_COLUMNS = MANUAL_COLUMNS = ['fecha', 'Code provincia alpha', 'Code comunidad autónoma alpha', 'casos_uci_target__ratio_population__num_casos__lag_-7', 'casos_uci__ratio_population__num_uci__lag_7', 'casos_uci__ratio_population__num_uci__lag_14', 'casos_uci__ratio_population__num_casos__lag_14', 'casos_uci__ratio_population__num_casos__lag_21', 'casos_uci__ratio_population__num_hosp__lag_14', 'casos__ratio_population__num_casos_prueba_pcr__lag_14', 'casos__ratio_population__num_casos_prueba_test_ac__lag_14', 'casos__ratio_population__num_casos_prueba_ag__lag_14', 'casos__ratio_population__num_casos_prueba_pcr__lag_21', 'casos__ratio_population__num_casos_prueba_test_ac__lag_21', 'casos__ratio_population__num_casos_prueba_ag__lag_21', 'googletrends__muerte__lag_10', 'googletrends__edema__lag_10', 'googletrends__cementerio__lag_10', 'googletrends__tanatorio__lag_10', 'googletrends__paliativos__lag_20', 'googletrends__entubado__lag_20', 'googletrends__enfermo terminal__lag_20', 'googletrends__coronavirus__lag_30', 'googletrends__confinamiento__lag_30', 'googletrends__enfermo__lag_30']
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
df_casos_uci_target = helpers.remove_location(df_casos_uci_target, 'sum')
### Death age groups ----
# df_casos_uci_ = df_casos_uci.drop(columns=[TARGET_VARIABLE_0])
df_casos_uci_lagged = helpers.shift_timeseries_by_lags(df_casos_uci, FIX_COLUMNS, LAG_UCI)
df_casos_uci_lagged = helpers.add_prefix(df_casos_uci_lagged, 'casos_uci__', FIX_COLUMNS)
df_casos_uci_lagged = helpers.remove_location(df_casos_uci_lagged, 'sum')
### Cases tested ----
df_casos_lagged = helpers.shift_timeseries_by_lags(df_casos, FIX_COLUMNS, LAG_CASOS)
df_casos_lagged = helpers.add_prefix(df_casos_lagged, 'casos__', FIX_COLUMNS)
df_casos_lagged = helpers.remove_location(df_casos_lagged, 'sum')
### AEMET temperature ----
df_aemet_lagged = helpers.shift_timeseries_by_lags(df_aemet, FIX_COLUMNS, LAG_OTHER)
df_aemet_lagged = helpers.add_prefix(df_aemet_lagged, 'aemet__', FIX_COLUMNS)
df_aemet_lagged = helpers.remove_location(df_aemet_lagged, 'mean')
### Google Trends ----
df_googletrends_lagged = helpers.shift_timeseries_by_lags(df_googletrends, FIX_COLUMNS, LAG_OTHER)
df_googletrends_lagged = helpers.add_prefix(df_googletrends_lagged, 'googletrends__', FIX_COLUMNS)
df_googletrends_lagged = helpers.remove_location(df_googletrends_lagged, 'sum')
### Movements to Madrid (`mitma`) ----
df_mitma_lagged = helpers.shift_timeseries_by_lags(df_mitma, FIX_COLUMNS, LAG_OTHER)
df_mitma_lagged = helpers.add_prefix(df_mitma_lagged, 'mitma__', FIX_COLUMNS)
df_mitma_lagged = helpers.remove_location(df_mitma_lagged, 'sum')
### Vaccination in Spain (`mscbs`) ----
df_mscbs_lagged = helpers.shift_timeseries_by_lags(df_mscbs, FIX_COLUMNS, LAG_OTHER)
df_mscbs_lagged = helpers.add_prefix(df_mscbs_lagged, 'mscbs__', FIX_COLUMNS)
df_mscbs_lagged = helpers.remove_location(df_mscbs_lagged, 'sum')
### Holidays ----
df_holidays_lagged = helpers.shift_timeseries_by_lags(df_holidays, FIX_COLUMNS, LAG_OTHER)
df_holidays_lagged = helpers.add_prefix(df_holidays_lagged, 'holidays__', FIX_COLUMNS)
df_holidays_lagged = helpers.remove_location(df_holidays_lagged, 'max')

# %%
## Calculate model for combination ----
def model_for_combination(combination, manual_column=None):

    # Merge the datasources
    df_merge = df_casos_uci_target.copy()
    if 'casos_uci' in combination:
        df_merge = df_merge.merge(df_casos_uci_lagged, on='fecha')
    if 'casos' in combination:
        df_merge = df_merge.merge(df_casos_lagged, on='fecha')
    if 'aemet' in combination:
        df_merge = df_merge.merge(df_aemet_lagged, on='fecha')
    if 'googletrends' in combination:
        df_merge = df_merge.merge(df_googletrends_lagged, on='fecha')
    if 'mitma' in combination:
        df_merge = df_merge.merge(df_mitma_lagged, on='fecha')
    if 'mscbs' in combination:
        df_merge = df_merge.merge(df_mscbs_lagged, on='fecha')
    if 'holidays' in combination:
        df_merge = df_merge.merge(df_holidays_lagged, on='fecha')

    # Remove `fecha`
    df_merge = df_merge.drop(columns=FIX_COLUMNS, errors='ignore')
    df_merge = df_merge.dropna().reset_index(drop=True)

    # Best business explanatory columns
    if manual_column is not None:
        df_merge = df_merge.loc[:, df_merge.columns.isin(manual_column)]

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

    # Ensure to remove all the non-significant variables
    for _ in range(0, 5):
        # Filter by `p-value`
        results_filtered = results.pvalues[results.pvalues.values<=P_VALUE]
        results_filtered_col = results_filtered.keys()

        # New model, after filter
        model = sm.OLS(y_train, X_train_scaled[results_filtered_col])
        results = model.fit()

    # Predict
    # y_train_pred = results.predict(X_train_scaled[results_filtered_col])
    y_test_pred = results.predict(X_test_scaled[results_filtered_col])
    # y_test_pred = results.predict(X_test_scaled)

    # Append results
    # results.rsquared_adj
    df_combination = pd.DataFrame({
        'Combination': str(combination), 
        'Combination length': len(combination), 
        'Number of features': len(results_filtered_col),
        # 'Number of features': X_test_scaled.shape[1], 
        'R2 train': results.rsquared_adj, 
        'R2 test': metrics.r2_score(y_test, y_test_pred)}, index=[0])

    df_coefficients = pd.DataFrame({
        'Combination': str(combination), 
        'Feature': results.pvalues.keys(),
        'Coefficient': results.params.values, 
        'p-value': results.pvalues.values})
        
    return df_combination, df_coefficients, combination, model, results, X_train_scaled, y_train, X_test_scaled, y_test

# %%
## All possible column combinations ----
all_combinations = []
for i in range(1, len(SOURCES)+1):
    all_combinations = all_combinations + list(itertools.combinations(SOURCES, i))

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
### Selection of some candidates ----
t1 = df_all_combinations[df_all_combinations['Combination length'] == 1]
t2 = df_all_combinations[(df_all_combinations['Combination length'] == 2) & (['googletrends' in x for x in df_all_combinations['Combination']])]
t3 = df_all_combinations[(df_all_combinations['Combination length'] == 3) & (['googletrends' in x for x in df_all_combinations['Combination']])]

# %%
### Coefficients per case ----
# df_all_coefficients_pivot[df_all_coefficients_pivot.index.isin(t1['Combination'])].dropna(how='all').T

# %%
## Coefficient plot ----
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# 'casos_uci', 'casos', 
df_combination, df_coefficients, combination, model, results, X_train_scaled, y_train, X_test_scaled, y_test = model_for_combination(('casos', 'googletrends'), MANUAL_COLUMNS)
df_coefficients = df_coefficients.sort_values('Coefficient', ascending=True)

plt.figure(figsize=[6,10])
sns.barplot(x='Coefficient', y='Feature', palette=colors_from_values(df_coefficients['p-value'], "YlOrRd"), data=df_coefficients)

# %%
## Correlation plot ----
def corrplot_simple(d, title):
    sns.set_theme(style="white")
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title(title)

# TODO: Think about a `zip` or `exec`
df_list = [df_casos_uci, df_casos_uci_lagged, df_casos_lagged, df_aemet_lagged, df_googletrends_lagged, df_mitma_lagged, df_mscbs_lagged, df_holidays_lagged]
[corrplot_simple(pd.concat([df_casos_uci_target, x], axis=1), None) for x in df_list]

# %%
## ProfileReport ----
ProfileReport_setup = {
    'samples': None,
    'correlations': None,
    'missing_diagrams': None,
    'duplicates': None,
    'interactions': None,
    'explorative': False,
}

# ProfileReport(df_casos_uci, title="df_casos_uci", **ProfileReport_setup).to_widgets()
# ProfileReport(df_casos, title="df_casos", **ProfileReport_setup).to_widgets()
# ProfileReport(df_aemet, title="df_aemet", **ProfileReport_setup).to_widgets()
# ProfileReport(df_googletrends, title="df_googletrends", **ProfileReport_setup).to_widgets()
# ProfileReport(df_mitma, title="df_mitma", **ProfileReport_setup).to_widgets()
# ProfileReport(df_mscbs, title="df_mscbs", **ProfileReport_setup).to_widgets()
# ProfileReport(df_holidays, title="df_holidays", **ProfileReport_setup).to_widgets()

# %%















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