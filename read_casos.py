# %%
import pandas as pd

# %%
province_cases = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_tecnica_provincia.csv', parse_dates=['fecha'])
province_cases = province_cases.dropna(subset=['provincia_iso'])

province_cases_uci = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv', parse_dates=['fecha'])
province_cases_uci = province_cases_uci.dropna(subset=['provincia_iso'])


province_code = pd.read_csv('data/ine.es/Province_Codigo.csv', sep='\t', converters = {'CODAUTO': str, 'CPRO': str}, keep_default_na=False)

df = province_cases.merge(province_code, left_on='provincia_iso', right_on='Code', how='left')

# %%
