# %%
# !pip install pytrends
# https://pypi.org/project/pytrends/

# %%
## Libraries ----
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
from joblib import dump, load
import helpers

# %%
## Testing ----
# kw_list = ["coronavirus", "covid", "confinamiento"]
# pytrends = TrendReq(hl='en-US', tz=360)  
# pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-01 2021-06-23', geo='ES-AN', gprop='')
# df = pytrends.interest_over_time()
# df.plot()

# %%
## Comunidad autónoma information ----
province_code = helpers.province_code()

# %%
## Get information from `Google Trends`
kw_lists = [["coronavirus", "covid", "confinamiento", "vacuna"],
            ["astrazeneca", "janssen", "infectados", "muerte"],
            ["temperatura", "fiebre", "edema"],
            ["pfizer", "hospital", "dolor", "tanatorio"]]
df_append = pd.DataFrame()
pytrends = TrendReq(hl='en-US', tz=360)  

for kw_list in kw_lists:
  for ca in province_code['Code comunidad autónoma alpha'].unique():
    geo_ = f'ES-{ca}'
    while True:
      try:
        pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-01 2021-06-23', geo=geo_, gprop='')
        df = pytrends.interest_over_time()
      except:
        print('retry', kw_list, geo_)
        continue
      else:
        df = df.reset_index()
        df['geo'] = geo_
        df['ca'] = ca
        df = pd.melt(df, id_vars=['date', 'geo', 'ca'], value_vars=kw_list)
        df_append = df_append.append(df, ignore_index=True)
        break

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