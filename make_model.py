# %%
## Libraries ----
import pandas as pd
import numpy as np
from datetime import datetime
import helpers
from joblib import dump, load

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
df_casos_uci_age = df_casos_uci_age.add_prefix('uci_age__')

# %%
### Cases tested ----
df_casos_pruebas = df_casos.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
df_casos_pruebas = pd.pivot_table(df_casos_pruebas, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['num_casos_prueba_pcr', 'num_casos_prueba_test_ac', 'num_casos_prueba_ag', 'num_casos_prueba_elisa', 'num_casos_prueba_desconocida'], aggfunc=np.sum, fill_value=0)

df_casos_pruebas.columns = ['__'.join(x) for x in df_casos_pruebas.columns]
df_casos_pruebas = df_casos_pruebas.reset_index()
df_casos_pruebas = df_casos_pruebas.add_prefix('tests__')

# %%
### AEMET temperature ----
df_aemet_pivot = df_aemet[[('tmed', 'mean'), ]]
df_aemet_pivot = df_aemet_pivot.reset_index()
df_aemet_pivot.columns = [x[0] for x in df_aemet_pivot.columns]
df_aemet_pivot = df_aemet_pivot.merge(province_code, on='Code provincia alpha')
df_aemet_pivot = pd.pivot_table(df_aemet_pivot, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['tmed'], aggfunc=np.sum, fill_value=0)

df_aemet_pivot.columns = ['__'.join(x) for x in df_aemet_pivot.columns]
df_aemet_pivot = df_aemet_pivot.reset_index()
df_aemet_pivot = df_aemet_pivot.add_prefix('aemet__')

# %%
### Google Trends ----
df_googletrends_pivot = df_googletrends
df_googletrends_pivot = df_googletrends_pivot.pivot(index=['date'], columns=['ca'])

df_googletrends_pivot.columns = ['__'.join(x) for x in df_googletrends_pivot.columns]
df_googletrends_pivot = df_googletrends_pivot.reset_index()
df_googletrends_pivot = df_googletrends_pivot.add_prefix('google_trends__')

# %%
### Movements to Madrid (`mitma`) ----
filter_ = df_mitma['destino_province']==TARGET_PROVINCE
df_mitma_filtered = df_mitma[filter_]
df_mitma_filtered = df_mitma_filtered.drop('destino_province', axis=1)
df_mitma_filtered = df_mitma_filtered.add_prefix('mitma__')

# %%
df_merge = df_casos_uci_num_defunciones.copy()
df_merge = df_merge.rename({'uci_defun__fecha': 'fecha'}, axis=1)

def merge_df_to_add(df_to_add, date_column, df_merge):
    df = df_merge.merge(df_to_add, left_on='fecha', right_on=date_column, how='outer')
    df = df.drop(date_column, axis=1)
    return df

df_merge = merge_df_to_add(df_casos_uci_age, 'uci_age__fecha', df_merge)
df_merge = merge_df_to_add(df_casos_pruebas, 'tests__fecha', df_merge)
df_merge = merge_df_to_add(df_aemet_pivot, 'aemet__fecha', df_merge)
df_merge = merge_df_to_add(df_googletrends_pivot, 'google_trends__date', df_merge)
df_merge = merge_df_to_add(df_mitma_filtered, 'mitma__fecha', df_merge)

# %%
# TODO:

# y = muertes
# lag 7, 14 para todas

# mitma
# ponerlo en columnas
# lag de 7 y 14

# falta el censo
# faltan las vacunas

# hacer un standardscaler

# Explain
# * https://gist.github.com/aigera2007/567a6d34cefb30c7c6255c20e40f24fb/raw/9c9cb058d1e00533b7dd9dc8f0fd9d3ad03caabb/corr_matrix.py
# * https://gist.github.com/aigera2007/789c2a996900bc88ea3f0d9494cbb9d0/raw/996fb3b7d0d46a151ab4486d841e9baeed30767b/roc_auc_curve.py

# SHAP: 
# * https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
# * https://aigerimshopenova.medium.com/random-forest-classifier-and-shap-how-to-understand-your-customers-and-interpret-a-black-box-model-6166d86820d9
