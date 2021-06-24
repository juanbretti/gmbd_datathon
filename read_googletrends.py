# %%
# !pip install pytrends
# https://pypi.org/project/pytrends/

# %%
## Libraries ----
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
from joblib import dump, load

# %%
## Testing ----
# kw_list = ["coronavirus", "covid", "confinamiento"]
# pytrends = TrendReq(hl='en-US', tz=360)  
# pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-01 2021-06-23', geo='ES-MD', gprop='')
# df = pytrends.interest_over_time()
# df.plot()

# %%
## Comunidad autónoma information ----
province_code = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'Code comunidad autónoma numérico': str, 'Code provincia numérico': str}, keep_default_na=False)

# %%
## Get information from `Google Trends`
kw_lists = [["coronavirus", "covid", "confinamiento", "vacuna", "pfizer"],
            ["temperatura", "fiebre", "edema", "dolor", "who"]]
            # ["astrazeneca", "janssen", "infectados", "muerte", "hospital"],
df_append = pd.DataFrame()
pytrends = TrendReq(hl='en-US', tz=360)  

for kw_list in kw_lists:
    for ca in province_code['Code comunidad autónoma alpha'].unique():
        geo_ = f'ES-{ca}'
        pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-01 2021-06-23', geo=geo_, gprop='')
        df = pytrends.interest_over_time()
        df = df.reset_index()
        df['geo'] = geo_
        df['ca'] = ca
        df = pd.melt(df, id_vars=['date', 'geo', 'ca'], value_vars=kw_list)
        df_append = df_append.append(df, ignore_index=True)

# %%

# %%
## Weekly to daily data ----
# https://stackoverflow.com/a/59856663/3780957

# Pivot the keywords
df1 = df_append.pivot(index=['date', 'geo', 'ca'], columns=['variable'], values=['value'])
columns_ = [x[1] for x in df1.columns.to_flat_index()]
df1.columns = columns_
df1 = df1.reset_index()

# Explode the weekly data into daily
df2 = (df1.set_index('date')  # Only can be dates
          .groupby('ca')[columns_]  # The rest of the indexes
          .resample('d')
          .ffill()
        #   .div(7)  # To divide by 7 the value
          .reset_index()
         )
df2

# %%
## Export ----
dump(df2, 'storage/df_export_googletrends.joblib') 

# %%