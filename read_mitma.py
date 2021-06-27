# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/meses-completos/202103_maestra1_mitma_distrito.tar
# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz

# %%
## Libraries ----
import pandas as pd
import numpy as np
from datetime import datetime
import helpers
from joblib import dump, load

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
province_code = province_code[['Code provincia numérico', 'Code provincia alpha']].drop_duplicates()

### Merge datasets ----
df_aggregate = df_aggregate.merge(province_code, left_on='origen_province', right_on='Code provincia numérico')
df_aggregate = df_aggregate.merge(province_code, left_on='destino_province', right_on='Code provincia numérico')
df_aggregate = df_aggregate.drop(['Code provincia numérico_x', 'Code provincia numérico_y', 'origen_province', 'destino_province'], axis=1)
df_aggregate = df_aggregate.rename(columns={'Code provincia alpha_x': 'origen_province', 'Code provincia alpha_y': 'destino_province'})

### Pivot ----
# df2_pivot = pd.pivot_table(df2, values=['viajes_sum', 'viajes_km_mean'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum, 'viajes_km_mean': np.mean}, fill_value=0)
df_pivot = pd.pivot_table(df_aggregate, values=['viajes__sum'], index=['fecha', 'destino_province'], columns=['origen_province'], aggfunc={'viajes__sum': np.sum}, fill_value=0)

### Replace column names ----
columns_ = [x[1] for x in df_pivot.columns.to_flat_index()]
df_pivot.columns = columns_
df_pivot = df_pivot.reset_index()

# %%
## Prepare for model ----
filter_ = df_pivot['destino_province']==helpers.target_province
df_pivot_filtered = df_pivot[filter_]
df_pivot_filtered = df_pivot_filtered.drop(columns='destino_province')
# Complete all the series
df_pivot_filtered = df_pivot_filtered.set_index('fecha').resample('d').ffill()
# Flatten column names and remove index
df_pivot_filtered = df_pivot_filtered.reset_index()

# %%
## Export ----
dump(df_pivot_filtered, 'storage/df_export_mitma.joblib') 

# %%