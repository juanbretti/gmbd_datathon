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

province_code = helpers.province_code()

# %%
# Cases tested
df_casos_pruebas = province_cases.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
df_casos_pruebas = pd.pivot_table(df_casos_pruebas, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['num_casos_prueba_pcr', 'num_casos_prueba_test_ac', 'num_casos_prueba_ag', 'num_casos_prueba_elisa', 'num_casos_prueba_desconocida'], aggfunc=np.sum, fill_value=0)
# Complete all the series
df_casos_pruebas = df_casos_pruebas.resample('d').ffill()
# Flatten column names and remove index
df_casos_pruebas.columns = ['__'.join(x) for x in df_casos_pruebas.columns]
df_casos_pruebas = df_casos_pruebas.reset_index()

# %%
dump(df_casos_pruebas, 'storage/df_export_casos.joblib') 
dump(province_cases_uci, 'storage/df_export_casos_uci.joblib') 

# %%
