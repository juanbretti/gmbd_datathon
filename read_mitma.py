# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/meses-completos/202103_maestra1_mitma_distrito.tar
# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz

# %%
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
from datetime import datetime
import helpers
from joblib import dump, load

# %%
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
# https://stackoverflow.com/questions/993358/creating-a-range-of-dates-in-python
# https://stackoverflow.com/a/26583750/3780957

date_start = '2020-02-21'
date_end = '2021-05-09'
# date_end = '2020-03-10'
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
        df2 = df2.reset_index()
        df2.columns = ['_'.join(x) for x in df2.columns.to_flat_index()]

        # Append
        df_aggregate = df_aggregate.append(df2, ignore_index=True)
    except:
        print('Error', date[0])

    # Temporary store
    if idx % 10 == 0:
        print('Store', idx)
        # dump(df_aggregate, 'storage/df_aggregate.joblib') 

# Final store
# dump(df_aggregate, 'storage/df_aggregate.joblib')

# %%
# df3 = df2.reset_index().merge(postal_codes.add_prefix('origen_'), left_on='origen_', right_on='origen_municipio_mitma')
# df3 = df3.merge(postal_codes.add_prefix('destino_'), left_on='destino_', right_on='destino_municipio_mitma')

# %%
df_aggregate = load('storage/df_aggregate.joblib')

# Source `Madrid`
# idx = df2['origen_province_']=='28'
# df2_pivot = pd.pivot_table(df2, values=['viajes_sum', 'viajes_km_mean'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum, 'viajes_km_mean': np.mean}, fill_value=0)
df_pivot = pd.pivot_table(df_aggregate, values=['viajes_sum'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum}, fill_value=0)
df_pivot.shape

df_pivot.reset_index()['fecha_'].value_counts()

# %%
postal_codes = pd.read_csv('https://postal.cat/download/postalcat.csv', sep=';', converters = {'cp': str})
municipio_mitma = pd.read_csv('data/mitma.gob.es/relaciones_municipio_mitma.csv', sep='|', converters = {'municipio': str, 'municipio_mitma': str})

# Remove the _AM
municipio_mitma['municipio_mitma_simple'] = municipio_mitma['municipio_mitma'].apply(lambda x: x.replace("_AM", ""))

# Keep one per `mitma`
municipio_mitma = municipio_mitma[['municipio_mitma', 'municipio_mitma_simple']].drop_duplicates()

postal_codes = postal_codes[['cp', 'provincia']].drop_duplicates()
postal_codes = postal_codes.merge(municipio_mitma, left_on='cp', right_on='municipio_mitma_simple')
postal_codes = postal_codes[['municipio_mitma', 'cp', 'provincia']]