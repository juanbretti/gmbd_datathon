# https://opendata.aemet.es/
# %%
## Libraries ----
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from joblib import dump, load
import helpers

# %%
## Helpers ----

key = 'xxx'
querystring = {"api_key": key}
headers = {'cache-control': "no-cache", 'Accept': 'text/plain',}

# Climatologicos
def get_climatologicos(start, end):
    start_ = start.strftime("%Y-%m-%d")
    end_ = end.strftime("%Y-%m-%d")

    url =f'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{start_}T00%3A00%3A00UTC/fechafin/{end_}T23%3A59%3A59UTC/todasestaciones'
    response = requests.request("GET", url, headers=headers, params=querystring)
    response_json = json.loads(response.text)
    response_url = requests.get(response_json['datos'])
    response_json_df = json.loads(response_url.text)
    df_climatologicos= pd.json_normalize(response_json_df)

    print('Request', start_, end_)

    return df_climatologicos

### For testing ----
# start = datetime(2020, 3, 17)
# end = datetime(2020, 3, 17)
# df = get_climatologicos(start, end)

# %%
## Get all the weather reports ----
# Covid range
date_start = datetime.strptime(helpers.start_date, "%Y-%m-%d")
date_end = datetime.strptime(helpers.end_date, "%Y-%m-%d")

# Creating ranges
dates_from = pd.date_range(date_start, date_end, freq='25D').tolist()
dates_to = pd.date_range(date_start + timedelta(days=24), date_end, freq='25D').tolist()
dates_to.append(date_end)
dates_all = pd.DataFrame({'From': dates_from, 'To': dates_to})

df_aggregate = pd.DataFrame()

for idx, date in dates_all.iterrows():
    try:
        df = get_climatologicos(date['From'], date['To'])
        df_aggregate = df_aggregate.append(df, ignore_index=True)
    except:
        print('Error', date['From'], date['To'])

    # Temporary store
    if idx % 10 == 0:
        print('Store', idx)
        # dump(df_aggregate, 'storage/df_temp_aemet.joblib')

# dump(df_aggregate, 'storage/df_temp_aemet.joblib')

# %%
## Estaciones ----
url = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones/'
response = requests.request("GET", url, headers=headers, params=querystring)
response_json = json.loads(response.text)
response_url = requests.get(response_json['datos'])
response_json_df = json.loads(response_url.text)
df_estaciones = pd.json_normalize(response_json_df)

province_code = helpers.province_code()

df_estaciones = df_estaciones.merge(province_code, left_on='provincia', right_on='Provincia aemet')

# %%
df_aggregate = load('storage/df_temp_aemet.joblib') 

## Format conversion ----
### Float ----
cols_to_float = ['altitud', 'tmed', 'prec', 'tmin', 'tmax', 'dir', 'velmedia', 'racha', 'sol', 'presMax', 'presMin']
def string_to_float(x):
    try:
        return x.str.replace(',', '.').astype(float)
    except:
        return np.nan
df_aggregate[cols_to_float] = df_aggregate[cols_to_float].apply(string_to_float, axis=1)

### Date ----
df_aggregate['fecha'] = pd.to_datetime(df_aggregate['fecha'], format="%Y-%m-%d", errors='coerce')

# %%
## Merge and groupby ----
df = df_aggregate.merge(df_estaciones, on='indicativo')
df_groupby = df.groupby(['fecha', 'Code provincia alpha']).agg({'tmed': ['mean', 'std', 'min', 'max'], 'prec': ['sum', 'mean', 'std', 'min', 'max'], 'tmin': ['mean', 'std', 'min', 'max'], 'tmax': ['mean', 'std', 'min', 'max']})

# %%
## Prepare data for model ----
df_groupby_filterd = df_groupby[[('tmed', 'mean'), ]]
df_groupby_filterd = df_groupby_filterd.reset_index()
df_groupby_filterd.columns = [x[0] for x in df_groupby_filterd.columns]
df_groupby_filterd = df_groupby_filterd.merge(province_code, on='Code provincia alpha')
df_groupby_filterd = pd.pivot_table(df_groupby_filterd, index=['fecha'], columns=['Code comunidad autónoma alpha'], values=['tmed'], aggfunc=np.mean, fill_value=0)
# Complete all the series
df_groupby_filterd = df_groupby_filterd.resample('d').interpolate(limit_direction='both')
# Flatten column names and remove index
df_groupby_filterd.columns = ['__'.join(x) for x in df_groupby_filterd.columns]
df_groupby_filterd = df_groupby_filterd.reset_index()

# %% 
dump(df_groupby_filterd, 'storage/df_export_aemet.joblib') 

# %%
## For the linear model ----
# df_lm = df_groupby[[('tmed', 'mean'), ('prec', 'sum'), ('tmin', 'min'), ('tmax', 'max')]]

df_lm1 = pd.pivot(df_groupby.reset_index(), index=['fecha'], columns=['Code provincia alpha'])
df_lm1 = df_lm1.resample('d').interpolate(limit_direction='both')
df_lm2 = df_lm1.reset_index()

df_lm3 = pd.melt(df_lm2, id_vars=['fecha'])
df_lm3 = df_lm3.rename({'variable_2': 'Code provincia alpha'}, axis=1)

df_lm4 = pd.pivot(df_lm3, index=['fecha', 'Code provincia alpha'], columns=['variable_0', 'variable_1'])
df_lm4.columns = ['__'.join(x) for x in df_lm4.columns]
df_lm4 = df_lm4.reset_index()

province_ = province_code[['Code comunidad autónoma alpha', 'Code provincia alpha']].drop_duplicates()
df_lm5 = df_lm4.merge(province_, on='Code provincia alpha')

dump(df_lm5, 'storage/df_export_aemet_lm.joblib') 
# %%
