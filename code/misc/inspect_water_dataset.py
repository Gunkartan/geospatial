import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../datasets/raw_water.csv')
    print(f'The total number of rows is {len(df)}')
    print(f'The total number of columns is {len(df.columns)}')
    print(f'There are {df.isna().sum()} NaN values per column')
    print(f'There are {np.isinf(df.select_dtypes(include=np.number)).sum()} infinite values per column')