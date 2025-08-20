
import numpy as np
import pandas as pd


def align_timestamps_stock_dataframes(
        stocks_data_dfs: list[pd.DataFrame],
        timestamps_as_index: bool=False
) -> pd.DataFrame:
    """Align multiple stock data DataFrames to have consistent timestamps.
    
    This function takes a list of stock data DataFrames and aligns their timestamps
    to ensure they all have the same time points, filling in missing data with
    forward-filled values.
    
    Args:
        stocks_data_dfs: List of stock data DataFrames to align
        timestamps_as_index: Whether to use timestamps as the DataFrame index (default: False)
        
    Returns:
        pd.DataFrame: List of aligned stock data DataFrames
    """
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
        stock_df.loc[:, 'TimeStamp'] = timedf.loc[(len(timedf)-len(stock_df)):, 'AllTime'].to_numpy()
        stock_df = stock_df.reset_index()
        stock_df = stock_df.drop(['index'], axis=1)
        if timestamps_as_index:
            stock_df.index = stock_df['TimeStamp']
        stocks_data_dfs[i] = stock_df

    return stocks_data_dfs


class InsufficientSharesException(Exception):
    """Exception raised when there are insufficient shares for a transaction."""
    pass
