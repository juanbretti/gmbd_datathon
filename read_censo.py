# %%
import pandas as pd
import numpy as np
import helpers
from joblib import dump, load
import dateparser

# %%
censo = pd.read_csv('https://www.ine.es/jaxiT3/files/t/es/csv_bd/31304.csv?nocab=1', sep='\t', thousands='.')

# %%
# Fix dates
dates = censo['Periodo'].unique()
dates_value = pd.Series(dates).apply(lambda x: dateparser.parse(x))
dates_df = pd.DataFrame({'Periodo': dates, 'Date': dates_value})

censo = censo.merge(dates_df, on='Periodo')

# %%
# Add province and comunidad autónoma information
censo['Code provincia numérico'] = censo['Provincias'].apply(lambda x: x[0:2])
province_code = helpers.province_code()
province_code = province_code[['Code comunidad autónoma alpha', 'Code provincia numérico']].drop_duplicates()

censo = censo.merge(province_code, on='Code provincia numérico')

# %%
# Filter and pivot
censo_filtered = censo[(censo['Date']>='2019-12-01') & (censo['Edad']=='Total') & (censo['Sexo']=='Ambos sexos')]
censo_pivot = pd.pivot_table(censo_filtered, index=['Date'], columns=['Code comunidad autónoma alpha'], values=['Total'], aggfunc=np.sum, fill_value=0)

# Extend end date
index = censo_pivot.index.tolist()
index[2] = pd.to_datetime(helpers.end_date)
censo_pivot.index = index

# %%
censo_2 = censo_pivot.resample('d').ffill()
censo_2.columns = [x[1] for x in censo_2.columns.to_flat_index()]

# %%
dump(censo_2, 'storage/df_export_censo.joblib') 

# %%
