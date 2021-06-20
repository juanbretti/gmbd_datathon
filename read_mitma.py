# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/meses-completos/202103_maestra1_mitma_distrito.tar
# https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz

# %%
import pandas as pd
import numpy as np
from datetime import datetime

# %%
# url = 'https://opendata-movilidad.mitma.es/maestra1-mitma-municipios/ficheros-diarios/2020-02/20200221_maestra_1_mitma_municipio.txt.gz'
# url = 'https://opendata-movilidad.mitma.es/maestra1-mitma-distritos/ficheros-diarios/2020-02/20200221_maestra_1_mitma_distrito.txt.gz'
# df = pd.read_csv(url, sep='|', decimal='.', parse_dates=['fecha'])

# %%
def read_mitma(date, detail='distrito'):
    # https://stackabuse.com/how-to-format-dates-in-python
    month_ = date.strftime("%Y-%m")
    day_ = date.strftime("%Y%m%d")
    url = f'https://opendata-movilidad.mitma.es/maestra1-mitma-{detail}s/ficheros-diarios/{month_}/{day_}_maestra_1_mitma_{detail}.txt.gz'
    print('Getting', url)
    df = pd.read_csv(url, sep='|', decimal='.', parse_dates=['fecha'])
    return df

# %%
date = datetime(2020, 2, 21)
df = read_mitma(date, detail='distrito')
df.groupby(['fecha', 'origen', 'destino']).agg({'viajes': ['sum', 'mean', 'std', 'min', 'max'], 'viajes_km': ['sum', 'mean', 'std', 'min', 'max']})
# %%
# https://stackoverflow.com/questions/993358/creating-a-range-of-dates-in-python
# https://stackoverflow.com/a/26583750/3780957

date1 = '2011-05-03'
date2 = '2011-05-10'
mydates = pd.date_range(date1, date2).tolist()
mydates[0].day

# %%
postal_codes = pd.read_csv('https://postal.cat/download/postalcat.csv', sep=';', converters = {'cp': str})

# %%
postal_codes['provincia'].value_counts()
# %%
