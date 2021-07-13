# https://stackoverflow.com/questions/17834995/how-to-convert-opendocument-spreadsheets-to-a-pandas-dataframe
# https://pypi.org/project/pandas-ods-reader/
# https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/vacunaCovid19.htm
# https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/documentos/Informe_Comunicacion_20210625.ods

# %%
## Libraries ----
from numpy.lib.histograms import histogram
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# tt = read_mscbs(datetime(2021,6,25))

# %%
## Loop ----
date_start = helpers.start_date
date_end = helpers.end_date
dates_all = pd.date_range(date_start, date_end)

# %%
df_aggregate = pd.DataFrame()
df_aggregate_sheet_name = pd.DataFrame()

for idx, date in pd.DataFrame(dates_all).iterrows():
    try:
        df = read_mscbs(date[0])

        # Getting the first sheet, which is more likely the appropriate one
        sheet_name_first = list(df)[0]
        sheet_first = df[sheet_name_first]
        sheet_first['date'] = date[0]
        df_aggregate = df_aggregate.append(sheet_first, ignore_index=True)

        # List of sheets names
        df_sheet_name = pd.DataFrame({'date': [date[0]], 'sheet_names': [str(list(df))], 'number_of_sheets': [len(list(df))]})
        df_aggregate_sheet_name = df_aggregate_sheet_name.append(df_sheet_name)
        
        # # Temporary store
        # if idx % 10 == 0:
        #     print('Store', idx)
        #     dump(df_aggregate, 'storage/df_temp_mscbs.joblib') 
        #     dump(df_aggregate_sheet_name, 'storage/df_temp_mscbs_sheet_name.joblib') 
    
    except:
        print('Error', date[0])

# Final store
# dump(df_aggregate, 'storage/df_temp_mscbs.joblib') 
# dump(df_aggregate_sheet_name, 'storage/df_temp_mscbs_sheet_name.joblib') 

# %%
## Exploration ----
# df_aggregate_sheet_name['sheet_names'].value_counts()

# df_aggregate['Unnamed: 0'].isna().value_counts()
# df_aggregate['Dosis administradas (2)'].isna().value_counts()
# df_aggregate.isna().sum()

# df_aggregate['Unnamed: 0'].value_counts()

# df_aggregate['Dosis administradas (2)'].hist()
# df_aggregate['Dosis administradas (2)'].describe()

# df_aggregate['date'].value_counts()

# %%
## Pivot and prepare ----
### Big dataframe ----
df_aggregate = load('storage/df_temp_mscbs.joblib')
# Some fixing
df_aggregate = df_aggregate.rename(columns={'Unnamed: 0': 'Comunidad Autónoma mscbs'})
df_aggregate['Comunidad Autónoma mscbs'] = df_aggregate['Comunidad Autónoma mscbs'].replace({'C. Valenciana*': 'C. Valenciana'})

### Province information ----
province_code = helpers.province_code()[['Comunidad Autónoma mscbs', 'Code comunidad autónoma alpha']].drop_duplicates()
df_aggregate = df_aggregate.merge(province_code, on='Comunidad Autónoma mscbs', how='inner')

# %%
## Add `Censo` data ----
### CONTROL: Remove this part
# Load `Censo` data
df_censo = load('storage/df_export_censo.joblib')

# External data
province_code = helpers.province_code()[['Code provincia alpha', 'Code comunidad autónoma alpha']].drop_duplicates()
df_censo = df_censo.merge(province_code, on='Code provincia alpha')
# Group by `Comunidad Autónoma`
df_censo = df_censo.groupby(['Code comunidad autónoma alpha', 'Date']).aggregate({'Total': np.sum})
df_censo = df_censo.reset_index()

# Add the `Censo` data
df_aggregate_censo = df_aggregate.merge(df_censo, left_on=['Code comunidad autónoma alpha', 'date'], right_on=['Code comunidad autónoma alpha', 'Date'])

# Calculate the `ratio_population`
df_aggregate_censo['ratio_population__Dosis administradas (2)'] = df_aggregate_censo['Dosis administradas (2)']/df_aggregate_censo['Total']*100e3

# %%
### Pivot ----
# df2_pivot = pd.pivot_table(df2, values=['viajes_sum', 'viajes_km_mean'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum, 'viajes_km_mean': np.mean}, fill_value=0)
### CONTROL: Change the `values` and `aggfunc`
df_pivot = pd.pivot_table(df_aggregate_censo, values=['ratio_population__Dosis administradas (2)'], index=['date'], columns=['Code comunidad autónoma alpha'], aggfunc={'ratio_population__Dosis administradas (2)': np.sum}, fill_value=0)
df_pivot = df_pivot.resample('d').interpolate(limit_direction='both').reset_index()
df_pivot = df_pivot.set_index('date')

# %%
### Create the `zero` vaccines rows for the dataframe
date_start = helpers.start_date
date_end = df_pivot.index.min() - timedelta(days=1)
dates_all = pd.date_range(date_start, date_end)

# Create a DataFrame with zeros
df_zero = pd.DataFrame(0, index=dates_all, columns=df_pivot.columns)
df_zero.index.name = 'date'
# Concatenate the zeros and the previous pivot
df_pivot_zero = pd.concat([df_zero, df_pivot], axis=0)

# %%
# https://stackoverflow.com/a/59383020/3780957
# Daily dosis
df_pivot_zero_diff = df_pivot_zero.diff()
df_pivot_zero_diff['date_diff'] = df_pivot_zero_diff.index.to_series().diff().dt.days
# Make daily values
df_pivot_zero_diff = df_pivot_zero_diff.apply(lambda x: x/df_pivot_zero_diff['date_diff'])
df_pivot_zero_diff = df_pivot_zero_diff.drop(columns='date_diff')
# Complete the time series
df_pivot_zero_diff = df_pivot_zero_diff.iloc[1:,].resample('d').interpolate(limit_direction='both').reset_index()
# df_pivot_zero_diff = df_pivot_zero_diff.dropna()

### Replace column names ----
def replace_columns(df, prefix):
    columns_ = [f'{prefix}_{x[1]}' for x in df.columns.to_flat_index()]
    df.columns = columns_
    return df.reset_index()

df_pivot_zero_flat = replace_columns(df_pivot_zero.copy(), 'cumulative')
df_pivot_zero_diff_flat = replace_columns(df_pivot_zero_diff.copy().set_index('date'), 'daily')

# %%
### Concatenate the two tables ----
df = pd.concat([df_pivot_zero_flat.set_index('date'), df_pivot_zero_diff_flat.set_index('date')], axis=1)

# %%
## Prepare for model ----
# Flatten column names and remove index
df_reseted = df.reset_index()

# %%
## Export ----
dump(df_reseted, 'storage/df_export_mscbs.joblib') 

# %%
## For the linear model ----
def prepare_for_lm(df, target_name):
    df = df.reset_index()
    df = pd.melt(df, id_vars=['date'])
    df = df.drop(columns=[None])
    df = df.rename({'value': target_name}, axis=1)
    return df

df_pivot_zero_1 = prepare_for_lm(df_pivot_zero, 'cumulative')
df_pivot_zero_diff_1 = prepare_for_lm(df_pivot_zero_diff, 'daily')

df_lm = df_pivot_zero_1.merge(df_pivot_zero_diff_1, on=['date', 'Code comunidad autónoma alpha'], how='outer')
df_lm = df_lm.dropna(subset=['Code comunidad autónoma alpha']) 
df_lm = df_lm[~(df_lm['Code comunidad autónoma alpha']=='')]

df_lm = df_lm.fillna(0)

province_ = province_code[['Code comunidad autónoma alpha', 'Code provincia alpha']].drop_duplicates()
df_lm1 = df_lm.merge(province_, on='Code comunidad autónoma alpha')

df_lm1 = df_lm1.rename({'date': 'fecha'}, axis=1)

dump(df_lm1, 'storage/df_export_mscbs_lm.joblib') 

# %%

