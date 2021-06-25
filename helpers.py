# %%
import pandas as pd

# %%
# https://stackoverflow.com/a/40941603/3780957
# https://stackoverflow.com/a/43034285/3780957

def shift_timeseries_by_lags(df, fix_columns, lag_numbers, lag_label='lag'):
    df_fix = df[fix_columns]
    df_lag = df.drop(columns=fix_columns)

    df_lagged = pd.concat({f'{lag_label}_{lag}':
        df_lag.shift(lag) for lag in lag_numbers},
        axis=1)
    df_lagged.columns = ['__'.join(reversed(x)) for x in df_lagged.columns.to_flat_index()]

    return pd.concat([df_fix, df_lagged], axis=1)

# df = shift_timeseries_by_lags(df_province_cases, fix_columns=['provincia_iso', 'fecha', 'num_casos', 'num_casos_prueba_pcr', 'Comunidad Autónoma'], lag_numbers=[1,2,3])
# df.columns.to_flat_index()

# %%
def province_code():
    df = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'Code comunidad autónoma numérico': str, 'Code provincia numérico': str}, keep_default_na=False)
    return df

#%%