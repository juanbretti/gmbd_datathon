# %%
import pandas as pd
import helpers
from joblib import dump, load

# %%
province_cases = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_tecnica_provincia.csv', parse_dates=['fecha'])
province_cases = province_cases.dropna(subset=['provincia_iso'])  # Only drop if `provincia_iso` is NaN

province_cases_uci = pd.read_csv('https://cnecovid.isciii.es/covid19/resources/casos_hosp_uci_def_sexo_edad_provres.csv', parse_dates=['fecha'])
province_cases_uci = province_cases_uci.dropna(subset=['provincia_iso'])  # Only drop if `provincia_iso` is NaN

# %%
dump(province_cases, 'storage/df_export_casos.joblib') 
dump(province_cases_uci, 'storage/df_export_casos_uci.joblib') 

# %%
