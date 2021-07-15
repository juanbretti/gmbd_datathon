# %%
import pandas as pd
import numpy as np
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
# https://gist.github.com/benjaminmgross/d71f161d48378d34b6970fa6d7378837
def press_statistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_
    https://statisticaloddsandends.wordpress.com/2018/07/30/the-press-statistic-for-linear-regression/
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )

    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - press / sst
 
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
def add_prefix(df, prefix, exclude):
    columns_all = pd.Series(df.columns)
    columns_to_prefix = pd.Series(columns_all).isin(exclude)
    columns_all[~columns_to_prefix] = prefix+columns_all[~columns_to_prefix]
    df.columns = columns_all
    return df