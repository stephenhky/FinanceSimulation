
from datetime import datetime, timedelta
from operator import itemgetter
from collections import defaultdict
import logging
import json

import numpy as np
import pandas as pd
from .portfolio import Portfolio
from ..data.preader import get_dividends_df


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
                idx = (start+end) // 2
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

        df = pd.concat(dataframes)
        df['TimeStamp'] = df['TimeStamp'].map(lambda ts: datetime.strftime(ts, '%Y-%m-%d'))
        return df

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


class DynamicPortfolioWithDividends(DynamicPortfolio):
    def __init__(self, symbol_nbshares, current_date, cash=0., cacheddir=None):
        # current_date is a string, of format '%Y-%m-%d', such as '2020-02-23'
        super(DynamicPortfolioWithDividends, self).__init__(symbol_nbshares, current_date, cacheddir=cacheddir)
        self.startcash = cash
        self.cashtimeseries = [{'date': current_date, 'cash': self.startcash}]

    def calculate_cash_from_dividends(self, enddate):
        startdate = self.timeseries[0]['date']
        self.cashtimeseries = [{'date': startdate, 'dividend': 0., 'cash': self.startcash}]

        for i, periodinfo in enumerate(self.timeseries):
            series_startdate = periodinfo['date']
            series_enddate = self.timeseries[i+1]['date'] if i < len(self.timeseries)-1 else enddate

            symbol_dividends_dataframes = []
            for symbol, nbshares in periodinfo['portfolio'].symbols_nbshares.items():
                dividend_df = get_dividends_df(symbol)
                if len(dividend_df) == 0:
                    continue
                dividend_df = dividend_df[(dividend_df['TimeStamp']>=series_startdate) & (dividend_df['TimeStamp']<series_enddate)].copy()
                if len(dividend_df) == 0:
                    continue
                dividend_df['Dividends'] *= nbshares
                symbol_dividends_dataframes.append(dividend_df)

            if len(symbol_dividends_dataframes) == 0:
                continue

            period_dividend_dataframe = pd.concat(symbol_dividends_dataframes)
            period_dividend_dataframe = period_dividend_dataframe.sort_values('TimeStamp')

            for dividend_date, dividend in zip(period_dividend_dataframe['TimeStamp'], period_dividend_dataframe['Dividends']):
                cash = self.cashtimeseries[-1]['cash']
                self.cashtimeseries.append({'date': dividend_date, 'dividend': dividend, 'cash': cash+dividend})

    def get_portfolio_values_overtime(self, startdate, enddate, cacheddir=None):
        worthdf = super(DynamicPortfolioWithDividends, self).get_portfolio_values_overtime(startdate, enddate, cacheddir=cacheddir)
        worthdf.rename(columns={'value': 'stock_value'}, inplace=True)
        self.calculate_cash_from_dividends(enddate)
        
        dividends_df = pd.DataFrame.from_records(self.cashtimeseries) if len(self.cashtimeseries) > 0 else pd.DataFrame(
            np.empty(0, dtype=[('date', 'S20'), ('dividend', 'f8'), ('cash', 'f8')])
        )
        dividends_df = dividends_df.rename(columns={'date': 'TimeStamp'})
        dividends_df = dividends_df.groupby('TimeStamp').aggregate({'dividend': 'sum', 'cash': 'max'}).reset_index()
        dividends_df = dividends_df.sort_values('TimeStamp')
        dividends_df['TimeStamp'] = dividends_df['TimeStamp'].astype(str)
        worthdf = worthdf.merge(dividends_df, on='TimeStamp', how='left')
        worthdf['cash'] = worthdf['cash'].ffill()
        worthdf = worthdf.fillna(0.)
        worthdf['value'] = worthdf['stock_value'] + worthdf['cash']

        return worthdf
