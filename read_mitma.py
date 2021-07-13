# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/meses-completos/202103_maestra1_mitma_distrito.tar
# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz

# %%
## Libraries ----
import pandas as pd
import numpy as np
from datetime import datetime
import helpers
from joblib import dump, load

## Constant ----
RATIO_PREFIX = 'ratio_population__'

# %%
## Download from `mitma` ----
# url = 'https://opendata-movilidad.mitma.es/maestra1-mitma-municipios/ficheros-diarios/2020-02/20200221_maestra_1_mitma_municipio.txt.gz'
# url = 'https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz'
# df = pd.read_csv(url, sep='|', decimal='.', parse_dates=['fecha'])

# %%
def read_mitma(date, detail='municipio'):
    """Download information

    Args:
        date (datetime): Date
        detail (str, optional): Detail to be downloaded. Defaults to 'distrito' ['distrito', 'municipio'].

    Returns:
        DataFrame: Downloaded information
    """
    # https://stackabuse.com/how-to-format-dates-in-python
    month_ = date.strftime("%Y-%m")
    day_ = date.strftime("%Y%m%d")
    url = f'https://opendata-movilidad.mitma.es/maestra1-mitma-{detail}s/ficheros-diarios/{month_}/{day_}_maestra_1_mitma_{detail}.txt.gz'
    print('Getting', url)
    df = pd.read_csv(url, sep='|', decimal='.', parse_dates=['fecha'])
    return df

# %%
## Loop ----
# https://stackoverflow.com/questions/993358/creating-a-range-of-dates-in-python
# https://stackoverflow.com/a/26583750/3780957

date_start = helpers.start_date
date_end = helpers.end_date
dates_all = pd.date_range(date_start, date_end)

# %%
df_aggregate = pd.DataFrame()

for idx, date in pd.DataFrame(dates_all).iterrows():
    try:
        # Download
        df = read_mitma(date[0], detail='municipio')
        
        # Group by
        df1 = df.copy()
        df1['origen_province'] = [x[0:2] for x in df['origen']]
        df1['destino_province'] = [x[0:2] for x in df['destino']]

        df2 = df1.groupby(['fecha', 'origen_province', 'destino_province']).agg({'viajes': ['sum', 'mean', 'std', 'min', 'max'], 'viajes_km': ['sum', 'mean', 'std', 'min', 'max']})

        # Change columns name
        df2.columns = ['__'.join(x) for x in df2.columns.to_flat_index()]
        df2 = df2.reset_index()

        # Append
        df_aggregate = df_aggregate.append(df2, ignore_index=True)
    except:
        print('Error', date[0])

    # Temporary store
    if idx % 10 == 0:
        print('Store', idx)
        dump(df_aggregate, 'storage/df_temp_mitma.joblib') 

# Final store
dump(df_aggregate, 'storage/df_temp_mitma.joblib')

# %%
## Pivot and prepare ----
### Big dataframe ----
df_aggregate = load('storage/df_temp_mitma.joblib')

### Province information ----
province_code = helpers.province_code()
province_code = province_code[['Code provincia numérico', 'Code provincia alpha', 'Code comunidad autónoma alpha']].drop_duplicates()

### Merge datasets ----
df_aggregate = df_aggregate.merge(province_code, left_on='origen_province', right_on='Code provincia numérico')
df_aggregate = df_aggregate.merge(province_code, left_on='destino_province', right_on='Code provincia numérico')
df_aggregate = df_aggregate.rename(columns={'Code comunidad autónoma alpha_x': 'origen_comunidad_autonoma', 'Code provincia alpha_y': 'destino_provincia'})

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
df_aggregate_censo = df_aggregate.merge(df_censo, left_on=['origen_comunidad_autonoma', 'fecha'], right_on=['Code comunidad autónoma alpha', 'Date'])

# Calculate the `ratio_population`
# df_aggregate_censo['ratio_population__viajes__sum'] = df_aggregate_censo['viajes__sum']/df_aggregate_censo['Total']*100e3
cases_value_columns = ['viajes__sum', 'viajes__mean', 'viajes__std', 'viajes__min', 'viajes__max', 'viajes_km__sum', 'viajes_km__mean', 'viajes_km__std', 'viajes_km__min', 'viajes_km__max']
df_aggregate_censo = df_aggregate_censo[cases_value_columns].apply(lambda x: x/df_aggregate_censo['Total']*100e3).add_prefix(RATIO_PREFIX)

# Concatenate
aggegate_columns = ['fecha', 'viajes__sum', 'viajes__mean', 'viajes__std', 'viajes__min', 'viajes__max', 'viajes_km__sum', 'viajes_km__mean', 'viajes_km__std', 'viajes_km__min', 'viajes_km__max', 'origen_comunidad_autonoma', 'destino_provincia']
df_aggregate_concat = pd.concat([df_aggregate[aggegate_columns], df_aggregate_censo], axis=1)

# %%
### Pivot ----
### CONTROL: Change the `values` and `aggfunc`
# df2_pivot = pd.pivot_table(df2, values=['viajes_sum', 'viajes_km_mean'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum, 'viajes_km_mean': np.mean}, fill_value=0)
df_pivot = pd.pivot_table(df_aggregate_concat, values=['ratio_population__viajes__sum'], index=['fecha', 'destino_provincia'], columns=['origen_comunidad_autonoma'], aggfunc={'ratio_population__viajes__sum': np.sum}, fill_value=0)

### Replace column names ----
columns_ = [x[1] for x in df_pivot.columns.to_flat_index()]
df_pivot.columns = columns_
df_pivot = df_pivot.reset_index()

# %%
## Prepare for model ----
filter_ = df_pivot['destino_provincia']==helpers.target_province
df_pivot_filtered = df_pivot[filter_]
df_pivot_filtered = df_pivot_filtered.drop(columns='destino_provincia')
# Complete all the series
df_pivot_filtered = df_pivot_filtered.set_index('fecha').resample('d').interpolate(limit_direction='both')
# Flatten column names and remove index
df_pivot_filtered = df_pivot_filtered.reset_index()

# %%
## Export ----
dump(df_pivot_filtered, 'storage/df_export_mitma.joblib') 

# TODO: Completar datos de tiempo
# first     2020-02-21 00:00:00
# last      2021-05-09 00:00:00

# %%
## For the linear model ----
df_lm1 = pd.pivot_table(df_aggregate_concat, index=['fecha'], columns=['destino_provincia'], aggfunc=np.sum, fill_value=0)
df_lm1 = df_lm1.resample('d').interpolate(limit_direction='both')
df_lm2 = df_lm1.reset_index()

df_lm3 = pd.melt(df_lm2, id_vars=['fecha'])
df_lm3 = df_lm3.rename({'destino_provincia': 'Code provincia alpha'}, axis=1)

df_lm4 = pd.pivot(df_lm3, index=['fecha', 'Code provincia alpha'], columns=[None])
df_lm4.columns = ['__'.join(x) for x in df_lm4.columns]
df_lm4 = df_lm4.reset_index()

province_ = province_code[['Code comunidad autónoma alpha', 'Code provincia alpha']].drop_duplicates()
df_lm5 = df_lm4.merge(province_, on='Code provincia alpha')

dump(df_lm5, 'storage/df_export_mitma_lm.joblib') 

# %%