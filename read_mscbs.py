# https://stackoverflow.com/questions/17834995/how-to-convert-opendocument-spreadsheets-to-a-pandas-dataframe
# https://pypi.org/project/pandas-ods-reader/
# https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/vacunaCovid19.htm
# https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/documentos/Informe_Comunicacion_20210625.ods

# %%
## Libraries ----
from numpy.lib.histograms import histogram
import pandas as pd
import numpy as np
from datetime import datetime
import helpers
from joblib import dump, load

# %%
## Download from `mitma` ----
def read_mscbs(date):
    day_ = date.strftime("%Y%m%d")
    url = f'https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/documentos/Informe_Comunicacion_{day_}.ods'
    print('Getting', url)
    df = pd.read_excel(url, engine='odf', sheet_name=None)
    return df

tt = read_mscbs(datetime(2021,6,25))

# %%
## Loop ----
date_start = '2020-02-21'
date_end = '2021-06-30'
dates_all = pd.date_range(date_start, date_end)

# %%
df_aggregate = pd.DataFrame()
df_aggregate_sheet_name = pd.DataFrame()

for idx, date in pd.DataFrame(dates_all).iterrows():
    try:
        df = read_mscbs(date[0])

        # Getting the first sheet, which is more likely the appropiate one
        sheet_name_first = list(df)[0]
        sheet_first = df[sheet_name_first]
        sheet_first['date'] = date[0]
        df_aggregate = df_aggregate.append(sheet_first, ignore_index=True)

        # List of sheets names
        df_sheet_name = pd.DataFrame({'date': [date[0]], 'sheet_names': [str(list(df))], 'number_of_sheets': [len(list(df))]})
        df_aggregate_sheet_name = df_aggregate_sheet_name.append(df_sheet_name)
    except:
        print('Error', date[0])

    # Temporary store
    if idx % 10 == 0:
        print('Store', idx)
        dump(df_aggregate, 'storage/df_temp_mscbs.joblib') 
        dump(df_aggregate_sheet_name, 'storage/df_temp_mscbs_sheet_name.joblib') 

# Final store
dump(df_aggregate, 'storage/df_temp_mscbs.joblib') 
dump(df_aggregate_sheet_name, 'storage/df_temp_mscbs_sheet_name.joblib') 

# %%
## Exploration ----
df_aggregate_sheet_name['sheet_names'].value_counts()

df_aggregate['Unnamed: 0'].isna().value_counts()
df_aggregate['Dosis administradas (2)'].isna().value_counts()
df_aggregate.isna().sum()

df_aggregate['Unnamed: 0'].value_counts()

df_aggregate['Dosis administradas (2)'].hist()
df_aggregate['Dosis administradas (2)'].describe()

df_aggregate['date'].value_counts()

# %%
## Pivot and prepare ----
### Big dataframe ----
df_aggregate = load('storage/df_temp_mscbs.joblib')
# Some fixing
df_aggregate = df_aggregate.rename(columns={'Unnamed: 0': 'Comunidad Autónoma mscbs'})
df_aggregate['Comunidad Autónoma mscbs'] = df_aggregate['Comunidad Autónoma mscbs'].replace({'C. Valenciana*': 'C. Valenciana'})

### Province information ----
province_code = helpers.province_code()
province_code = province_code[['Comunidad Autónoma mscbs', 'Code comunidad autónoma alpha']].drop_duplicates()

### Merge datasets ----
df_aggregate = df_aggregate.merge(province_code, on='Comunidad Autónoma mscbs', how='inner')

# %%
### Pivot ----
# df2_pivot = pd.pivot_table(df2, values=['viajes_sum', 'viajes_km_mean'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum, 'viajes_km_mean': np.mean}, fill_value=0)
df_pivot = pd.pivot_table(df_aggregate, values=['Dosis administradas (2)'], index=['date'], columns=['Code comunidad autónoma alpha'], aggfunc={'Dosis administradas (2)': np.sum}, fill_value=0)
df_pivot = df_pivot.resample('d').ffill().reset_index()
df_pivot = df_pivot.set_index('date')

# https://stackoverflow.com/a/59383020/3780957
# Daily dosis
df_pivot_diff = df_pivot.diff()
df_pivot_diff['date_diff'] = df_pivot_diff.index.to_series().diff().dt.days
# Make daily values
df_pivot_diff = df_pivot_diff.apply(lambda x: x/df_pivot_diff['date_diff'])
df_pivot_diff = df_pivot_diff.drop(columns='date_diff')
# Complete the time series
df_pivot_diff = df_pivot_diff.resample('d').ffill().reset_index()
# df_pivot_diff = df_pivot_diff.dropna()

### Replace column names ----
def replace_columns(df, prefix):
    columns_ = [f'{prefix}_{x[1]}' for x in df.columns.to_flat_index()]
    df.columns = columns_
    return df.reset_index()

df_pivot = replace_columns(df_pivot, 'cumulative')
df_pivot_diff = replace_columns(df_pivot_diff.set_index('date'), 'daily')

# %%
### Concatenate the two tables ----
df = pd.concat([df_pivot.set_index('date'), df_pivot_diff.set_index('date')], axis=1)

# %%
## Export ----
dump(df, 'storage/df_export_mscbs.joblib') 

# %%