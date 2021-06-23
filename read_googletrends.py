# %%
# !pip install pytrends
# https://pypi.org/project/pytrends/

# %%
## Libraries ----
from pytrends.request import TrendReq
import pandas as pd

# %%
## Testing ----
# kw_list = ["coronavirus", "covid", "confinamiento"]
# pytrends = TrendReq(hl='en-US', tz=360)  
# pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-01 2021-06-23', geo='ES-MD', gprop='')
# df = pytrends.interest_over_time()
# df.plot()

# %%
## Comunidad autónoma information ----
province_code = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'CODAUTO': str, 'CPRO': str}, keep_default_na=False)

# %%
## Get information from `Google Trends`
kw_list = ["coronavirus", "covid", "confinamiento"]
pytrends = TrendReq(hl='en-US', tz=360)  
df_aggregate = pd.DataFrame()

for ca in province_code['Code comunidad autónoma alpha'].unique():
    geo_ = f'ES-{ca}'
    pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-01 2021-06-23', geo=geo_, gprop='')
    df = pytrends.interest_over_time()
    df = df.reset_index()
    df['geo'] = geo_
    df['ca'] = ca
    df_aggregate = df_aggregate.append(df, ignore_index=True)

# %%
df_aggregate

# %%
## Weekly to daily data ----
# https://stackoverflow.com/a/59856663/3780957

df1 = (df_aggregate.set_index('date')
          .groupby('geo')['coronavirus', 'covid', 'confinamiento']
          .resample('d')
          .ffill()
          .div(7)
          .reset_index()
         )

df1
# %%

df_aggregate['date'].describe(datetime_is_numeric=True)