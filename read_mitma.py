# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/meses-completos/202103_maestra1_mitma_distrito.tar
# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz

# %%
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
from datetime import datetime
import helpers

# %%
# url = 'https://opendata-movilidad.mitma.es/maestra1-mitma-municipios/ficheros-diarios/2020-02/20200221_maestra_1_mitma_municipio.txt.gz'
# url = 'https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz'
# df = pd.read_csv(url, sep='|', decimal='.', parse_dates=['fecha'])

# %%
def read_mitma(date, detail='distrito'):
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

date1 = '2021-05-03'
date2 = '2021-05-10'
mydates = pd.date_range(date1, date2).tolist()
mydates[0].day

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

# %%
# date = datetime(2020, 2, 21)
date = mydates[0]
df = read_mitma(date, detail='municipio')

# Extract the province number
df1 = df.copy()
df1['origen_province'] = [x[0:2] for x in df['origen']]
df1['destino_province'] = [x[0:2] for x in df['destino']]

df2 = df1.groupby(['fecha', 'origen_province', 'destino_province']).agg({'viajes': ['sum', 'mean', 'std', 'min', 'max'], 'viajes_km': ['sum', 'mean', 'std', 'min', 'max']})

df2 = df2.reset_index()
df2.columns = ['_'.join(x) for x in df2.columns.to_flat_index()]

# df3 = df2.reset_index().merge(postal_codes.add_prefix('origen_'), left_on='origen_', right_on='origen_municipio_mitma')
# df3 = df3.merge(postal_codes.add_prefix('destino_'), left_on='destino_', right_on='destino_municipio_mitma')

# %%
# Source `Madrid`
# idx = df2['origen_province_']=='28'
df2_pivot = pd.pivot_table(df2, values=['viajes_sum', 'viajes_km_mean'], index=['fecha_', 'origen_province_'], columns=['destino_province_'], aggfunc={'viajes_sum': np.sum, 'viajes_km_mean': np.mean}, fill_value=0)
df2_pivot.shape