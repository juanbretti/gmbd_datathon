# %%
# !pip install gdeltdoc --user
# !pip install urllib3

# %%

from gdeltdoc import GdeltDoc, Filters

f = Filters(
    keyword = "climate change",
    start_date = "2020-05-10",
    end_date = "2020-05-11"
)

gd = GdeltDoc()

# Search for articles matching the filters
articles = gd.article_search(f)

# Get a timeline of the number of articles matching the filters
timeline = gd.timeline_search("timelinevol", f)

#%%
from gdeltdoc import Filters, near, repeat

f = Filters(
    start_date = "2020-01-01",
    end_date = "2021-06-21",
    num_records = 250,
    keyword = "covid",
    country = "RU",
    # theme = ["TAX_DISEASE_CORONAVIRUS", "TAX_DISEASE_CORONAVIRUSES", "TAX_DISEASE_CORONAVIRUS_INFECTIONS"],
    near = near(10, "salud", "confinamiento", "muerte"),
    repeat = repeat(5, "planet")
)

gd = GdeltDoc()

timeline = gd.timeline_search("timelinevol", f)

# %%
timeline.plot(x='datetime', y='Volume Intensity')

# %%
import matplotlib.pyplot as plt

plt.scatter(timeline['datetime'], timeline['Volume Intensity'])
plt.show()
# %%
import requests
import pandas as pd
import io

# YYYYMMDDHHMMSS 
# 20200101000000
# 20210621235959
# url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=%22covid%20OR%20coronavirus%22&mode=timelinevol&STARTDATETIME=20200101000000&ENDDATETIME=20210621235959&FORMAT=csv'
# url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=coronavirus&mode=timelinevol&STARTDATETIME=20200101000000&ENDDATETIME=20210621235959&FORMAT=csv&sourcecountry=SP'
# url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=covid&mode=timelinevol&STARTDATETIME=20200101000000&ENDDATETIME=20210621235959&FORMAT=csv&sourcecountry=SP'
# url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=covid&mode=Timeline&STARTDATETIME=20200101000000&ENDDATETIME=20210621235959&FORMAT=csv&sourcecountry=SP'

def gdelt_search(x):
    url = f'https://api.gdeltproject.org/api/v2/doc/doc?query=%22{x}%22%20sourcecountry:SP&mode=timelinevol&timespan=18m&format=CSV'
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    return df

# %%
# ax = df.groupby('Series')['Value'].plot(kind='kde', legend=True)

# %%
# import seaborn as sns
# fig = sns.kdeplot(data=df, kind='kde', x='Date', y='Value', hue='Series')

# %%
df = gdelt_search('confinamiento')
df.plot(x='Date', y='Value')
# %%
