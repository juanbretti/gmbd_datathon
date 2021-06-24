# %%
import pandas as pd
import helpers
from joblib import dump, load

# %%
province_cases = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_tecnica_provincia.csv', parse_dates=['fecha'])
province_cases = province_cases.dropna(subset=['provincia_iso'])

province_cases_uci = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv', parse_dates=['fecha'])
province_cases_uci = province_cases_uci.dropna(subset=['provincia_iso'])

province_code = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'Code comunidad autónoma numérico': str, 'Code provincia numérico': str}, keep_default_na=False)

df_province_cases = province_cases.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')
df_province_cases_uci = province_cases_uci.merge(province_code, left_on='provincia_iso', right_on='Code provincia alpha')

# %%
dump(df_province_cases, 'storage/df_export_casos.joblib') 
dump(df_province_cases_uci, 'storage/df_export_casos_uci.joblib') 

# %%
helpers.shift_timeseries_by_lags(df_province_cases, 
    fix_columns=['provincia_iso', 'fecha', 'Code comunidad autónoma numérico', 'Comunidad Autónoma', 'Code provincia numérico', 'Provincia', 'Code', 'aemet'], 
    lag_numbers=[0,1,2,3])

# %^%