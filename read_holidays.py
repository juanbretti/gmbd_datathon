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
    df_provincia = pd.DataFrame({'Province': province, 'Date': holidays_, 'Holiday': 1})
    df_aggregate = df_aggregate.append(df_provincia, ignore_index=True)

df_aggregate['Date'] = pd.to_datetime(df_aggregate['Date'])

# %%
provinces = helpers.province_code()[['Code provincia alpha', 'Code comunidad autónoma alpha']].drop_duplicates()
df_merged = df_aggregate.merge(provinces, left_on='Province', right_on='Code provincia alpha')
df = pd.pivot_table(df_merged, index=['Date'], columns=['Code comunidad autónoma alpha'], values=['Holiday'], aggfunc=np.max, fill_value=0)
df = df.resample('d').asfreq().fillna(0)

# %%
# Flatten column names and remove index
df.columns = ['__'.join(x) for x in df.columns]
df = df.reset_index()

# %% 
dump(df, 'storage/df_export_holidays.joblib') 

# %%