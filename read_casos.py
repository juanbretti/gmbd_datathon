# %%
import pandas as pd
import numpy as np
import helpers
from joblib import dump, load

# %%
province_cases = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_tecnica_provincia.csv', parse_dates=['fecha'])
province_cases = province_cases.dropna(subset=['provincia_iso'])  # Only drop if `provincia_iso` is NaN

province_cases_uci = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv', parse_dates=['fecha'])
province_cases_uci = province_cases_uci.dropna(subset=['provincia_iso'])  # Only drop if `provincia_iso` is NaN

# Load `Province` data
province_code = helpers.province_code()[['Code comunidad autónoma alpha', 'Code provincia alpha']].drop_duplicates()
# Load `Censo` data
df_censo = load('storage/df_export_censo.joblib')

# Constants
RATIO_PREFIX = 'ratio_population__'

# %%
# Cases tested
df_cases = province_cases.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
df_cases_uci = province_cases_uci.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')

# Merge `Censo` data
df_cases = df_cases.merge(df_censo, left_on=['Code provincia alpha', 'fecha'], right_on=['Code provincia alpha', 'Date'])
df_cases_uci = df_cases_uci.merge(df_censo, left_on=['Code provincia alpha', 'fecha'], right_on=['Code provincia alpha', 'Date'])

# UCI, merge ages
df_cases_uci['grupo_edad_merged'] = df_cases_uci['grupo_edad'].replace({'0-9': '0-59', '10-19': '0-59', '20-29': '0-59', '30-39': '0-59', '40-49': '0-59', '50-59': '0-59'})

# Columns to normalize
cases_value_columns = ['num_casos', 'num_casos_prueba_pcr', 'num_casos_prueba_test_ac', 'num_casos_prueba_ag', 'num_casos_prueba_elisa', 'num_casos_prueba_desconocida']
cases_uci_value_columns = ['num_casos', 'num_hosp', 'num_uci', 'num_def']
cases_base_columns = ['fecha', 'provincia_iso', 'Code comunidad autónoma alpha']
cases_uci_base_columns = ['fecha', 'provincia_iso', 'Code comunidad autónoma alpha', 'grupo_edad_merged']
censo_column = 'Total'

# Calculate the ratio
# df_cases_ratio = df_cases[cases_value_columns].apply(lambda x: x/df_cases[censo_column]*100e3).add_prefix(RATIO_PREFIX)
# df_cases_uci_ratio = df_cases_uci[cases_uci_value_columns].apply(lambda x: x/df_cases_uci[censo_column]*100e3).add_prefix(RATIO_PREFIX)
df_cases_ratio = df_cases[cases_value_columns].apply(lambda x: x).add_prefix(RATIO_PREFIX)
df_cases_uci_ratio = df_cases_uci[cases_uci_value_columns].apply(lambda x: x).add_prefix(RATIO_PREFIX)

# Concatenate the tables
### CONTROL: Add here the original columns, by removing `cases_base_columns`
df_cases_concat = pd.concat([df_cases], axis=1)
df_cases_uci_concat = pd.concat([df_cases_uci], axis=1)

# %%
cases_value_columns_ratio = [RATIO_PREFIX+x for x in cases_value_columns]
cases_uci_value_columns_ratio = [RATIO_PREFIX+x for x in cases_uci_value_columns]

# Pivot the data values
### CONTROL: Add the list of columns `cases_value_columns`
df_cases_pivot = pd.pivot_table(df_cases_concat, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=cases_value_columns, aggfunc=np.sum, fill_value=0)
df_cases_uci_pivot = pd.pivot_table(df_cases_uci_concat, index=['fecha'], columns=['Code comunidad autónoma alpha', 'grupo_edad_merged'], values=cases_uci_value_columns, aggfunc=np.sum, fill_value=0)

# %%
# Target variable
# TODO: Normalize in 100k
filter_ = df_cases_uci['provincia_iso']==helpers.target_province
df_casos_uci_num_defunciones = df_cases_uci[filter_].groupby(['fecha']).aggregate({'num_def': np.sum})
df_casos_uci_num_defunciones = df_casos_uci_num_defunciones.resample('d').asfreq().fillna(0).reset_index()

# %%
# Flatten column names and remove index
def flatten(df):
    df = df.resample('d').asfreq().fillna(0)
    df.columns = ['__'.join(x) for x in df.columns]
    df = df.reset_index()
    return df

df_cases_pivot = flatten(df_cases_pivot)
df_cases_uci_pivot = flatten(df_cases_uci_pivot)

# %%
dump(df_cases_pivot, 'storage/df_export_cases.joblib') 
dump(df_cases_uci_pivot, 'storage/df_export_cases_uci.joblib') 
dump(df_casos_uci_num_defunciones, 'storage/df_export_cases_uci_num_defunciones.joblib') 

# %%
## For the linear model ----
df_cases_lm = df_cases_concat
df_cases_lm = df_cases_lm.rename({'provincia_iso': 'Code provincia alpha'}, axis=1)
df_cases_uci_lm = df_cases_uci_concat.groupby(['fecha', 'provincia_iso', 'Code comunidad autónoma alpha']).sum().reset_index()
df_cases_uci_lm = df_cases_uci_lm.rename({'provincia_iso': 'Code provincia alpha'}, axis=1)

dump(df_cases_lm, 'storage/df_export_cases_lm.joblib') 
dump(df_cases_uci_lm, 'storage/df_export_cases_uci_lm.joblib') 

# %%