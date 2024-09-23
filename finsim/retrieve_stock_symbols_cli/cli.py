
from argparse import ArgumentParser
import os

import pandas as pd
from ..data.finnhub import FinnHubStockReader


def get_argparser():
    argparser = ArgumentParser(description='Retrieve stock symbols from Finnhub')
    argparser.add_argument('outputpath', help='path of the stock symbols (*.json, *.h5, *.xlsx, *.csv)')
    argparser.add_argument('--finnhubtokenpath', help='path of Finnhub tokens')
    argparser.add_argument('--useenvtoken', help='Use the environment variable FINNHUBTOKEN as the tokens')
    argparser.add_argument('--shorten', default=False, action='store_true', help='shorten list of symbols')
    return argparser


def main_cli():
    # parsing argument
    args = get_argparser().parse_args()
    extension = os.path.splitext(args.outputpath)[-1]

    # check if the output directory exists
    dirname = os.path.dirname(args.outputpath)
    if not os.path.isdir(dirname):
        raise FileNotFoundError('Directory {} does not exist!'.format(dirname))

    # get Finnhub tokens
    if args.useenvtoken:
        finnhub_token = os.getenv('FINNHUBTOKEN')
        if finnhub_token is None:
            raise ValueError('Finnhub tokens not found in the environment variable $FINNHUBTOKEN.')
    else:
        finnhub_token = open(args.finnhubtokenpath, 'r').read().strip()

    # initialize FinnHub reader
    finnreader = FinnHubStockReader(finnhub_token)

    # grab symbols
    allsym = finnreader.get_all_US_symbols()
    allsymdf = pd.DataFrame(allsym)

    if args.shorten:
        filtered_symdf = allsymdf[allsymdf['mic'].isin(['XNAS', 'XNYS', 'ARCX'])]
        filtered_symdf = filtered_symdf[~filtered_symdf['type'].isin(['PUBLIC'])]
        filtered_symdf = filtered_symdf[~filtered_symdf['symbol'].str.contains('\.')]
        allsymdf = filtered_symdf

    if extension == '.h5':
        allsymdf.to_hdf(args.outputpath, 'fintable')
    elif extension == '.json':
        allsymdf.to_json(args.outputpath, orient='records')
    elif extension == '.xlsx':
        allsymdf.to_excel(args.outputpath)
    elif extension == '.csv':
        allsymdf.to_csv(args.outputpath)
    elif extension == '.pickle' or extension == '.pkl':
        allsymdf.to_pickle(args.outputpath)
    else:
        raise IOError('Extension {} not recognized.'.format(extension))