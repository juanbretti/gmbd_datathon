import pandas as pd
import numpy as np

s = pd.Series([np.nan, 99, 1, np.nan, 3])
s.interpolate(limit_direction='both')
