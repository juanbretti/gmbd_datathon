# https://opendata.aemet.es/centrodedescargas/ejemProgramas?

# %%
key = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqdWFuYnJldHRpK2FlbWV0QHN0dWRlbnQuaWUuZWR1IiwianRpIjoiZGI5OWRjNGQtY2VjZS00MjQ3LTk5ZTAtMjZiYTk4ODdkYTIwIiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE2MjQyMTQyMjIsInVzZXJJZCI6ImRiOTlkYzRkLWNlY2UtNDI0Ny05OWUwLTI2YmE5ODg3ZGEyMCIsInJvbGUiOiIifQ.Qx9OhUKCfMOaHLUlZnzwpQpWimdQXbQ4p-vbMolX_qc'

# %%
import requests

# url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones/"
url = "https://opendata.aemet.es/opendata/api/observacion/convencional/todas"

querystring = {"api_key":key}
headers = {
    'cache-control': "no-cache",
    'Accept': 'text/plain',
    }
response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)

# %%
import json
import pandas as pd

response_json = json.loads(response.text)
response_url = requests.get(response_json['datos'])
response_json_df = json.loads(response_url.text)
df = pd.json_normalize(response_json_df)

df.head().T

# %%

df.shape


# %%

# %%
import http.client

conn = http.client.HTTPSConnection("opendata.aemet.es")

headers = {
    'cache-control': "no-cache"
    }

conn.request("GET", f"/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones/?api_key={key}", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))


# %%

import urllib.request, json 
with urllib.request.urlopen(y['datos']) as url:
    data = json.loads(url.read().decode())
    print(data)

# %%

from urllib2 import Request, urlopen
import json

import pandas as pd    

request=Request('http://maps.googleapis.com/maps/api/elevation/json?locations='+path1+'&sensor=false')
response = urlopen(request)
elevations = response.read()
data = json.loads(elevations)
df = pd.json_normalize(data['results'])

# %%
# curl -X GET --header 'Accept: text/plain' 'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/2021-06-10T12%3A00%3A00UTC/fechafin/2021-06-10T13%3A00%3A00UTC/todasestaciones'
# https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/2021-06-10T12%3A00%3A00UTC/fechafin/2021-06-10T13%3A00%3A00UTC/todasestaciones
# https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/2021-06-10T00%3A00%3A00UTC/fechafin/2021-06-10T23%3A59%3A59UTC/todasestaciones
# curl -X GET -H "Cache-Control: no-cache" https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones/?api_key=jyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqbW9udGVyb2dAYWVtZXQuZXMiLCJqdGkiOiI3NDRiYmVhMy02NDEyLTQxYWMtYmYzOC01MjhlZWJlM2FhMWEiLCJleHAiOjE0NzUwNTg3ODcsImlzcyI6IkFFTUVUIiwiaWF0IjoxNDc0NjI2Nzg3LCJ1c2VySWQiOiI3NDRiYmVhMy02NDEyLTQxYWMtYmYzOC01MjhlZWJlM2FhMWEiLCJyb2xlIjoiIn0.xh3LstTlsP9h5cxz3TLmYF4uJwhOKzA0B6-vH8lPGGw
