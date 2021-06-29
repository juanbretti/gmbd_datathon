# %%
import pandas as pd
from sklearn import metrics

# %%
start_date = '2020-01-01'
start_date_vaccination = '2021-01-01' # Considering only vaccination time
end_date = '2021-06-26'
target_province = 'M'

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
def pct_change_by_lags(df, fix_columns, lag_numbers, lag_label='pct_change'):
    df = df.set_index(fix_columns)

    df_concat = pd.concat({f'{lag_label}_{lag}':
        df.pct_change(periods=lag, freq='D') for lag in lag_numbers},
        axis=1)

    df_concat.columns = ['__'.join(reversed(x)) for x in df_concat.columns.to_flat_index()]

    df_out = pd.concat([df, df_concat], axis=1)
    return df_out.reset_index()

# %%
def province_code():
    df = pd.read_csv('data/Province_Codigo.csv', sep='\t', converters = {'Code comunidad autónoma numérico': str, 'Code provincia numérico': str}, keep_default_na=False)
    return df

#%%
def timer(start_time=None):
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# %%
def metrics_custom(y_true, y_pred):
    print(f'Parson R2: {metrics.r2_score(y_true, y_pred)}')
    print(f'Mean Squared Error: {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'Mean Absolute Percengage Error: {metrics.mean_absolute_percentage_error(y_true, y_pred)}')

def metrics_custom2(y_train, y_train_pred, y_test, y_test_pred):
    print('** Train **')
    metrics_custom(y_train, y_train_pred)
    print('\n** Test **')
    metrics_custom(y_test, y_test_pred)

# %%