
import numpy as np
import pandas as pd


def align_timestamps_stock_dataframes(stocks_data_dfs, timestamps_as_index=False):
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
        stock_df.loc[:, 'TimeStamp'] = timedf.loc[(len(timedf)-len(stock_df)):, 'AllTime'].ravel()
        stock_df = stock_df.reset_index()
        stock_df = stock_df.drop(['index'], axis=1)
        if timestamps_as_index:
            stock_df.index = stock_df['TimeStamp']
        stocks_data_dfs[i] = stock_df

    return stocks_data_dfs