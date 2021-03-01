
from datetime import datetime, timedelta
from operator import itemgetter
from collections import defaultdict
import logging
import json

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

    def move_cursor_to_date(self, newdate):
        # date is a string, of format '%Y-%m-%d', such as '2020-02-23'
        self.timeseriesidx = self.find_cursor_for_date(newdate)
        self.symbols_nbshares = self.timeseries[self.timeseriesidx]['portfolio'].symbols_nbshares.copy()
        self.current_date = newdate

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
            if raise_insufficient_stock_error:
                errorfound = False
                for symbol, nbshares in sell_stocks.items():
                    if nbshares > self.symbols_nbshares[symbol]:
                        logging.error('Insufficient stock {} ({} shares, but want to sell {} shares)'.format(symbol, self.symbols_nbshares[symbol], nbshares))
                        errorfound = True
                if errorfound:
                    raise InsufficientSharesException('Insufficient stocks.')
            else:
                for symbol, nbshares in sell_stocks.items():
                    if nbshares > self.symbols_nbshares[symbol]:
                        sell_stocks[symbol] = self.symbols_nbshares[symbol]

        for symbol, nbshares in sell_stocks.items():
            symbols_nbshares[symbol] -= nbshares

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

    def generate_dynamic_portfolio_dict(self):
        dynport_dict = {'name': 'DynamicPortfolio'}
        dynport_dict['current_date'] = self.current_date
        dynport_dict['timeseries'] = []
        for portdict in self.timeseries:
            trade_date = portdict['date']
            portfolio = portdict['portfolio']
            dynport_dict['timeseries'].append({
                'date': trade_date,
                'portfolio': portfolio.symbols_nbshares
            })
        return dynport_dict

    def save_to_json(self, fileobj):
        dynport_dict = self.generate_dynamic_portfolio_dict()
        json.dump(dynport_dict, fileobj)

    def dumps_json(self):
        dynport_dict = self.generate_dynamic_portfolio_dict()
        return json.dumps(dynport_dict)

    @classmethod
    def load_from_dict(cls, dynportdict, cacheddir=None):
        assert dynportdict['name'] == 'DynamicPortfolio'
        dynport = cls(
            dynportdict['timeseries'][0]['portfolio'],
            dynportdict['timeseries'][0]['date'],
            cacheddir=cacheddir
        )
        for portdict in dynportdict['timeseries'][1:]:
            tradedate = portdict['date']
            symbols_nbshares = portdict['portfolio']
            dynport.timeseries.append({'date': tradedate,
                                       'portfolio': Portfolio(symbols_nbshares, cacheddir=cacheddir)
                                       })

        dynport.move_cursor_to_date(dynportdict['current_date'])
        return dynport
    
    @classmethod
    def load_from_json(cls, fileobj, cacheddir=None):
        dynportinfo = json.load(fileobj)
        return cls.load_from_dict(dynportinfo, cacheddir=cacheddir)