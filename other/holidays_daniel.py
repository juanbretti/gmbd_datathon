# -*- coding: utf-8 -*-
"""holidays.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uWggzvpFlEAUCyYFhxULbYn7nwge6aU3
"""
# %%
import datetime
import holidays
import pandas as pd

# %%
# pip install holidays

provinces = ['AN', 'AR','AS', 'CB', 'CL', 'CM', 'CN', 'CT', 'EX', 'GA', 'IB', 'MC', 'MD', 'NC', 'PV', 'RI', 'VC']

esp_holidays = {}

for prov in provinces:
    esp_holidays[prov] = list(holidays.CountryHoliday('ESP', years=[2020, 2021], prov=prov).keys())
        
print(esp_holidays)

start_date = datetime.datetime(year=2020, month=1, day=1)
end_date = datetime.datetime(year=2021, month=12, day=31)

df = pd.DataFrame()
df['Dates'] = pd.date_range(start_date, end_date)
df.head()

for prov in provinces:
  for index, val in enumerate(df['Dates']):
    df.loc[index, prov] = str(val).split()[0] in [date_obj.strftime('%Y-%m-%d') for date_obj in esp_holidays[prov]]

df.head(10)
# %%
