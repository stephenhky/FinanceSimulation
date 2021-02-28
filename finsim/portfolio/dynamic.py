
from datetime import datetime, timedelta
from operator import itemgetter
from collections import defaultdict
import logging

import pandas as pd
from .portfolio import Portfolio


class InsufficientSharesException(Exception):
    pass


class DynamicPortfolio(Portfolio):
    def __init__(self, symbol_nbshares, current_date, cacheddir=None):
        # current_date is a string, of format '%Y-%m-%d', such as '2020-02-23'
        super(DynamicPortfolio, self).__init__(symbol_nbshares, cacheddir=cacheddir)
        self.current_date = current_date
        self.timeseries = [{
            'date': self.current_date,
            'portfolio': Portfolio(self.symbols_nbshares, self.cacheddir)
        }]
        self.timeseriesidx = 0

    def sort_time_series(self):
        self.timeseries = sorted(self.timeseries, key=itemgetter('date'))

    def is_sorted(self):
        for i, j in zip(range(len(self.timeseries)-1), range(1, len(self.timeseries))):
            if not self.timeseries[i]['date'] < self.timeseries[j]['date']:
                return False
        return True

    def find_cursor_for_date(self, date):
        # date is a string, of format '%Y-%m-%d', such as '2020-02-23'
        start = 0
        end = len(self.timeseries)
        idx = (end-start) // 2

        if date < self.timeseries[0]['date']:
            return (-1)
        if date >= self.timeseries[end-1]['date']:
            return (end-1)

        found = False
        while not found:
            if date >= self.timeseries[idx]['date'] and (idx >= len(self.timeseries) or date < self.timeseries[idx+1]['date']):
                found = True
            else:
                if self.timeseries[idx]['date'] < date:
                    start = idx
                elif self.timeseries[idx]['date'] >= date:
                    end = idx
                idx = (end-start) // 2
        return idx

    def move_cursor_to_date(self, date):
        # date is a string, of format '%Y-%m-%d', such as '2020-02-23'
        self.timeseriesidx = self.find_cursor_for_date(date)
        self.symbols_nbshares = self.timeseries[self.timeseriesidx]['portfolio'].symbols_nbshares.copy()
        logging.debug('MOVED!')

    def trade(
            self,
            trade_date,
            buy_stocks={},
            sell_stocks={},
            check_valid=False,
            raise_insufficient_stock_error=False
    ):
        assert self.is_sorted()
        assert trade_date > self.timeseries[-1]['date']

        # rectifying
        overlapping_symbols = set(buy_stocks.keys()).intersection(set(sell_stocks.keys()))
        for symbol in overlapping_symbols:
            if buy_stocks[symbol] > sell_stocks[symbol]:
                buy_stocks[symbol] -= sell_stocks[symbol]
                del sell_stocks[symbol]
            elif buy_stocks[symbol] < sell_stocks[symbol]:
                sell_stocks[symbol] -= buy_stocks[symbol]
                del buy_stocks[symbol]
            else:
                del buy_stocks[symbol]
                del sell_stocks[symbol]

        symbols_nbshares = defaultdict(lambda : 0, buy_stocks)
        if check_valid:
            logging.debug('CHECK VALID')
            logging.debug(self.symbols_nbshares)
            if raise_insufficient_stock_error:
                errorfound = False
                for symbol, nbshares in sell_stocks.items():
                    logging.debug('{}: {} < {}'.format(symbol, nbshares, self.symbols_nbshares[symbol]))
                    logging.debug(nbshares > self.symbols_nbshares[symbol])
                    if nbshares > self.symbols_nbshares[symbol]:
                        logging.error('Insufficient stock {} ({} shares, but want to sell {} shares)'.format(symbol, self.symbols_nbshares[symbol], nbshares))
                        errorfound = True
                if errorfound:
                    raise InsufficientSharesException('Insufficient stocks.')
            else:
                for symbol, nbshares in sell_stocks.items():
                    logging.debug('{}: {} < {}'.format(symbol, nbshares, self.symbols_nbshares[symbol]))
                    logging.debug(nbshares > self.symbols_nbshares[symbol])
                    if nbshares > self.symbols_nbshares[symbol]:
                        logging.debug('UPDATE sell_stocks')
                        sell_stocks[symbol] = self.symbols_nbshares[symbol]

        for symbol, nbshares in sell_stocks.items():
            symbols_nbshares[symbol] -= nbshares
        # logging.debug(symbols_nbshares)

        difference_portfolio = Portfolio(symbols_nbshares, cacheddir=self.cacheddir)
        self.timeseries.append({
            'date': trade_date,
            'portfolio': self.timeseries[-1]['portfolio'] + difference_portfolio
        })

        self.move_cursor_to_date(trade_date)
        
    def get_portfolio_value(self, datestr):
        idx = self.find_cursor_for_date(datestr)
        return self.timeseries[idx]['portfolio'].get_portfolio_value(datestr)

    def get_portfolio_values_overtime(self, startdate, enddate, cacheddir=None):
        assert self.is_sorted()

        startidx = self.find_cursor_for_date(startdate)
        endidx = self.find_cursor_for_date(enddate)

        assert startidx <= endidx

        dataframes = []
        for portidx in range(startidx, endidx+1):
            thisport_startdate = max(startdate, self.timeseries[portidx]['date'])
            if portidx == endidx:
                thisport_enddate = enddate
            else:
                ed = datetime.strptime(self.timeseries[portidx+1]['date'], '%Y-%m-%d') - timedelta(1)
                thisport_enddate = datetime.strftime(ed, '%Y-%m-%d')
            dataframes.append(
                self.timeseries[portidx]['portfolio'].get_portfolio_values_overtime(
                    thisport_startdate,
                    thisport_enddate,
                    cacheddir=self.cacheddir
                )
            )

        return pd.concat(dataframes)
