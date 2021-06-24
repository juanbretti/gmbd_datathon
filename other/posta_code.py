# df3 = df2.reset_index().merge(postal_codes.add_prefix('origen_'), left_on='origen_', right_on='origen_municipio_mitma')
# df3 = df3.merge(postal_codes.add_prefix('destino_'), left_on='destino_', right_on='destino_municipio_mitma')



postal_codes = pd.read_csv('https://postal.cat/download/postalcat.csv', sep=';', converters = {'cp': str})
municipio_mitma = pd.read_csv('data/mitma.gob.es/relaciones_municipio_mitma.csv', sep='|', converters = {'municipio': str, 'municipio_mitma': str})

# Remove the _AM
municipio_mitma['municipio_mitma_simple'] = municipio_mitma['municipio_mitma'].apply(lambda x: x.replace("_AM", ""))

# Keep one per `mitma`
municipio_mitma = municipio_mitma[['municipio_mitma', 'municipio_mitma_simple']].drop_duplicates()

postal_codes = postal_codes[['cp', 'provincia']].drop_duplicates()
postal_codes = postal_codes.merge(municipio_mitma, left_on='cp', right_on='municipio_mitma_simple')
postal_codes = postal_codes[['municipio_mitma', 'cp', 'provincia']]