import os

import click
import pandas as pd
from ..data.finnhub import FinnHubStockReader


@click.command(help='Retrieve stock symbols from Finnhub')
@click.argument('outputpath')
@click.option('--finnhubtokenpath', default=None, help='path of Finnhub tokens')
@click.option('--useenvtoken', is_flag=True, default=False, help='Use the environment variable FINNHUBTOKEN as the tokens')
@click.option('--shorten', is_flag=True, default=False, help='shorten list of symbols')
def main_cli(outputpath, finnhubtokenpath=None, useenvtoken=False, shorten=False):
    """Main CLI function to retrieve stock symbols from Finnhub and save them to a file.
    
    This function parses command line arguments, retrieves stock symbols from Finnhub,
    optionally filters them, and saves them to a file in various formats.
    """
    extension = os.path.splitext(outputpath)[-1]

    # check if the output directory exists
    dirname = os.path.dirname(outputpath)
    if dirname and not os.path.isdir(dirname):
        raise FileNotFoundError('Directory {} does not exist!'.format(dirname))

    # get Finnhub tokens
    if useenvtoken:
        finnhub_token = os.getenv('FINNHUBTOKEN')
        if finnhub_token is None:
            raise ValueError('Finnhub tokens not found in the environment variable $FINNHUBTOKEN.')
    else:
        if finnhubtokenpath is None:
            raise ValueError('Either --finnhubtokenpath or --useenvtoken must be specified.')
        with open(finnhubtokenpath, 'r') as f:
            finnhub_token = f.read().strip()

    # initialize FinnHub reader
    finnreader = FinnHubStockReader(finnhub_token)

    # grab symbols
    allsym = finnreader.get_all_US_symbols()
    allsymdf = pd.DataFrame(allsym)

    if shorten:
        filtered_symdf = allsymdf[allsymdf['mic'].isin(['XNAS', 'XNYS', 'ARCX'])]
        filtered_symdf = filtered_symdf[~filtered_symdf['type'].isin(['PUBLIC'])]
        filtered_symdf = filtered_symdf[~filtered_symdf['symbol'].str.contains(r'\.')]
        allsymdf = filtered_symdf

    if extension == '.h5':
        allsymdf.to_hdf(outputpath, key='fintable')
    elif extension == '.json':
        allsymdf.to_json(outputpath, orient='records')
    elif extension == '.xlsx':
        allsymdf.to_excel(outputpath)
    elif extension == '.csv':
        allsymdf.to_csv(outputpath)
    elif extension == '.pickle' or extension == '.pkl':
        allsymdf.to_pickle(outputpath)
    else:
        raise IOError('Extension {} not recognized.'.format(extension))


if __name__ == '__main__':
    main_cli()