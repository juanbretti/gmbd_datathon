# %%
import pandas as pd
import helpers

# %%
province_cases = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_tecnica_provincia.csv', parse_dates=['fecha'])
province_cases = province_cases.dropna(subset=['provincia_iso'])

province_cases_uci = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv', parse_dates=['fecha'])
province_cases_uci = province_cases_uci.dropna(subset=['provincia_iso'])

province_code = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'CODAUTO': str, 'CPRO': str}, keep_default_na=False)

df_province_cases = province_cases.merge(province_code, left_on='provincia_iso', right_on='Code')
df_province_cases_uci = province_cases_uci.merge(province_code, left_on='provincia_iso', right_on='Code')

# %%
helpers.shift_timeseries_by_lags(df_province_cases, 
    fix_columns=['provincia_iso', 'fecha', 'CODAUTO', 'Comunidad Autónoma', 'CPRO', 'Provincia', 'Code', 'aemet'], 
    lag_numbers=[0,1,2,3])

# %%

df_province_cases.columns