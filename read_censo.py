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
province_code = province_code[['Code provincia numérico', 'Code provincia alpha']].drop_duplicates()
censo_ca = censo.merge(province_code, on='Code provincia numérico')

# %%
# Filter and pivot
censo_filtered = censo_ca[(censo_ca['Date']>='2019-12-01') & (censo_ca['Edad']=='Total') & (censo_ca['Sexo']=='Ambos sexos')]
censo_filtered = censo_filtered[['Date', 'Total', 'Code provincia numérico', 'Code provincia alpha']]

# Extend end date
censo_filtered['Date'] = censo_filtered['Date'].replace({censo_filtered['Date'].max(): pd.to_datetime(helpers.end_date)})

censo_2 = (censo_filtered
    .set_index('Date')
    .groupby(['Code provincia numérico', 'Code provincia alpha'])[['Total', ]]
    .resample('d')
    .interpolate(limit_direction='both')
)

censo_2 = censo_2.reset_index()
dump(censo_2, 'storage/df_export_censo.joblib') 

# %%

censo_2.groupby('Date').sum()