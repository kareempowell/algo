class algo(object):

  #choose a group of bitcoin, indices, and forex
  #iterate through the instruments using the select function
  #for each instrument measure their relative performance and store in a list
  #combine this list in a matrix and provide the securities with the lowest correlation and highest sharpe ratio
  #return this preferred list of securities
  #then select the securities based on the above
  #create a function to place orders for each of these securities. batch ordering?
  
  def select_instrument(self):
    #selecting instruments N.B. 1st line previously data = oanda.get_history
    data = api.get_history(
        instrument='EUR_USD',
        start='2025-02-06',
        end='2025-02-07',
        granularity='M1',
        price='M'
    )
    data.info()

  def measure_performance(self):
    #Backtesting: Build out momentum strategy | NB. Why is this backtesting?
    import numpy as np
    data['returns'] = np.log(data['c'] / data['c'].shift(1))
    cols = []
    for momentum in [15, 30, 60, 120, 150]:
        col = f'p_{momentum}'
        data[col] = np.sign(data['returns'].rolling(momentum).mean())
        cols.append(col)
    
    #visualize strategy performance | N.B. line 2 previously 'seaborn'
    from pylab import plt
    plt.style.use('seaborn-v0_8-colorblind')
    strats = ['returns']
    for col in cols:
        strat = f's_{col[2:]}'
        data[strat] = data[col].shift(1) * data['returns']
        strats.append(strat)
    data[strats].dropna().cumsum().apply(np.exp).plot(cmap='coolwarm');

#initialize automation
import pandas as pd
class MomentumTrader(tpqoa.tpqoa):
    def __init__(self, config_file, momentum):
        super(MomentumTrader, self).__init__(config_file)
        self.momentum = momentum
        self.min_length = momentum + 1
        self.position = 0
        self.units = 10000
        self.tick_data = pd.DataFrame()
    def on_success(self, time, bid, ask):
        trade = False
        # print(self.ticks, end=' ')
        self.tick_data = self.tick_data.append(
            pd.DataFrame({'b': bid, 'a': ask, 'm': (ask + bid) / 2},
               index=[pd.Timestamp(time).tz_localize(tz=None)])
        )
        self.data = self.tick_data.resample('5s', label='right').last().ffill()
        self.data['r'] = np.log(self.data['m'] / self.data['m'].shift(1))
        self.data['m'] = self.data['returns'].rolling(self.momentum).mean()
        self.data.dropna(inplace=True)
        if len(self.data) > self.min_length:
            self.min_length += 1
            if self.data['m'].iloc[-2] > 0 and self.position in [0, -1]:
              o = api.create_order(self.stream_instrument,
                             units=(1 - self.position) * self.units,
                             suppress=True, ret=True)
              print('\n*** GOING LONG ***')
              api.print_transactions(tid=int(o['id']) - 1)
              self.position = 1
        if self.data['m'].iloc[-2] < 0 and self.position in [0, 1]:
              o = api.create_order(self.stream_instrument,
                            units=-(1 + self.position) * self.units,
                            suppress=True, ret=True)
              print('\n*** GOING SHORT ***')
              self.print_transactions(tid=int(o['id']) - 1)
              self.position = -1
#may need to leave this for jupityer
  mt = MomentumTrader('/content/drive/MyDrive/Paueru/Projects/Models/2. AlgoTrading Models/oanda.cfg', momentum=5)
  mt.stream_data('EUR_USD', stop=100)

#initialize close out class
class closetrades(tpqoa.tpqoa):
  def closingactiveorders(self):
      #[3c]
      #closing out the final position. shows the complete, detailed order object
      from pprint import pprint
      o = mt.create_order('EUR_USD', units=-mt.position * mt.units,
                          suppress=True, ret = True)
      print('\n*** POSITION CLOSED ***')
      mt.print_transactions(tid=int(o['id']) - 1)
      print('\n')
      pprint(o)
    
          
  #source: https://github.com/GJason88/backtrader-backtests/blob/master/StochasticSR/Stochastic_SR_Backtest.py
  
  import backtrader as bt
  import logging
  import datetime
  import os.path
  import sys

class StochasticSR(bt.Strategy):
    '''Trading strategy that utilizes the Stochastic Oscillator indicator for oversold/overbought entry points, 
    and previous support/resistance via Donchian Channels as well as a max loss in pips for risk levels.'''
    # parameters for Stochastic Oscillator and max loss in pips
    # Donchian Channels to determine previous support/resistance levels will use the given period as well
    # http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Stochastic.php5 for Stochastic Oscillator formula and description
    params = (('period', 14), ('pfast', 3), ('pslow', 3), ('upperLimit', 80), ('lowerLimit', 20), ('stop_pips', .002))

    def __init__(self):
        '''Initializes logger and variables required for the strategy implementation.'''
        # initialize logger for log function (set to critical to prevent any unwanted autologs, not using log objects because only care about logging one thing)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(message)s', level=logging.CRITICAL, handlers=[
            logging.FileHandler("LOG.log"),
            logging.StreamHandler()
            ])

        self.order = None
        self.donchian_stop_price = None
        self.price = None
        self.stop_price = None
        self.stop_donchian = None

        self.stochastic = bt.indicators.Stochastic(self.data, period=self.params.period, period_dfast=self.params.pfast, period_dslow=self.params.pslow, 
        upperband=self.params.upperLimit, lowerband=self.params.lowerLimit)

    def log(self, txt, doprint=True):
        '''logs the pricing, orders, pnl, time/date, etc for each trade made in this strategy to a LOG.log file as well as to the terminal.'''
        date = self.data.datetime.date(0)
        time = self.data.datetime.time(0)
        if (doprint):
            logging.critical(str(date) + ' ' + str(time) + ' -- ' + txt)


    def notify_trade(self, trade):
        '''Run on every next iteration, logs the P/L with and without commission whenever a trade is closed.'''
        if trade.isclosed:
            self.log('CLOSE -- P/L gross: {}  net: {}'.format(trade.pnl, trade.pnlcomm))


    def notify_order(self, order):
        '''Run on every next iteration, logs the order execution status whenever an order is filled or rejected, 
        setting the order parameter back to None if the order is filled or cancelled to denote that there are no more pending orders.'''
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log('BUY -- units: 10000  price: {}  value: {}  comm: {}'.format(order.executed.price, order.executed.value, order.executed.comm))
                self.price = order.executed.price
            elif order.issell():
                self.log('SELL -- units: 10000  price: {}  value: {}  comm: {}'.format(order.executed.price, order.executed.value, order.executed.comm))
                self.price = order.executed.price
        elif order.status in [order.Rejected, order.Margin]:
            self.log('Order rejected/margin')
        
        self.order = None


    def stop(self):
        '''At the end of the strategy backtest, logs the ending value of the portfolio as well as one or multiple parameter values for strategy optimization purposes.'''
        self.log('(period {}) Ending Value: {}'.format(self.params.period, self.broker.getvalue()), doprint=True)


    def next(self):
        '''Checks to see if Stochastic Oscillator, position, and order conditions meet the entry or exit conditions for the execution of buy and sell orders.'''
        if self.order:
            # if there is a pending order, don't do anything
            return
        if self.position.size == 0:
            # When stochastic crosses back below 80, enter short position.
            if self.stochastic.lines.percD[-1] >= 80 and self.stochastic.lines.percD[0] <= 80:
                # stop price at last support level in self.params.period periods
                self.donchian_stop_price = max(self.data.high.get(size=self.params.period))
                self.order = self.sell()
                # stop loss order for max loss of self.params.stop_pips pips
                self.stop_price = self.buy(exectype=bt.Order.Stop, price=self.data.close[0]+self.params.stop_pips, oco=self.stop_donchian)
                # stop loss order for donchian SR price level
                self.stop_donchian = self.buy(exectype=bt.Order.Stop, price=self.donchian_stop_price, oco=self.stop_price)
            # when stochastic crosses back above 20, enter long position.
            elif self.stochastic.lines.percD[-1] <= 20 and self.stochastic.lines.percD[0] >= 20:
                # stop price at last resistance level in self.params.period periods
                self.donchian_stop_price = min(self.data.low.get(size=self.params.period))
                self.order = self.buy()
                # stop loss order for max loss of self.params.stop_pips pips
                self.stop_price = self.sell(exectype=bt.Order.Stop, price=self.data.close[0]-self.params.stop_pips, oco=self.stop_donchian)
                # stop loss order for donchian SR price level
                self.stop_donchian = self.sell(exectype=bt.Order.Stop, price=self.donchian_stop_price, oco=self.stop_price) 
  
        if self.position.size > 0:
            # When stochastic is above 70, close out of long position
            if (self.stochastic.lines.percD[0] >= 70):
                self.close(oco=self.stop_price)
        if self.position.size < 0:
            # When stochastic is below 30, close out of short position
            if (self.stochastic.lines.percD[0] <= 30):
                self.close(oco=self.stop_price)
    
