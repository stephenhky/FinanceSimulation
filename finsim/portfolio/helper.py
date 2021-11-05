
import numpy as np
import pandas as pd


def align_timestamps_stock_dataframes(stocks_data_dfs):
    # unify the timestamps columns
    timestampset = set()
    for stock_df in stocks_data_dfs:
        timestampset = timestampset.union(stock_df['TimeStamp'])
    alltimestamps = sorted(list(timestampset))
    timedf = pd.DataFrame({'AllTime': alltimestamps})

    # wrangle the stock dataframes
    for i, stock_df in enumerate(stocks_data_dfs):
        stock_df = pd.merge(stock_df, timedf, how='right', left_on='TimeStamp', right_on='AllTime').ffill()
        stock_df = stock_df.loc[~np.isnan(stock_df['TimeStamp']), stock_df.columns[:-1]]
        stocks_data_dfs[i] = stock_df.reset_index()

    return stocks_data_dfs