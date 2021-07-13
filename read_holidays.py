# -*- coding: utf-8 -*-
# %%
import datetime
import holidays
import pandas as pd
import numpy as np
import helpers
from joblib import dump, load

# pip install holidays

# %%
provinces = helpers.province_code()['Code provincia alpha'].drop_duplicates()

# %%

df_aggregate = pd.DataFrame()
for province in provinces:
    holidays_ = list(holidays.CountryHoliday('ESP', years=[2020, 2021], prov=province))
    df_provincia = pd.DataFrame({'Code provincia alpha': province, 'fecha': holidays_, 'Holiday': 1})
    df_aggregate = df_aggregate.append(df_provincia, ignore_index=True)

df_aggregate['fecha'] = pd.to_datetime(df_aggregate['fecha'])

# %%
provinces = helpers.province_code()[['Code provincia alpha', 'Code comunidad autónoma alpha']].drop_duplicates()
df_merged = df_aggregate.merge(provinces, left_on='Code provincia alpha', right_on='Code provincia alpha')
df = pd.pivot_table(df_merged, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['Holiday'], aggfunc=np.max, fill_value=0)
df = df.resample('d').asfreq().fillna(0)

# %%
# Flatten column names and remove index
df2 = df.copy()
df2.columns = ['__'.join(x) for x in df2.columns]
df2 = df2.reset_index()

# %% 
dump(df2, 'storage/df_export_holidays.joblib') 

# %%
## For the linear model ----
df_lm1 = pd.pivot(df_merged, index=['fecha'], columns=['Code provincia alpha', 'Code comunidad autónoma alpha'], values=['Holiday'])
df_lm1 = df_lm1.resample('d').asfreq().fillna(0)
df_lm2 = df_lm1.reset_index()

df_lm3 = pd.melt(df_lm2, id_vars=['fecha'])
df_lm3 = df_lm3.drop(columns=[None])

dump(df_lm3, 'storage/df_export_holidays_lm.joblib') 

# %%