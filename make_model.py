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

province_code = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'Code comunidad autónoma numérico': str, 'Code provincia numérico': str}, keep_default_na=False)

# %%