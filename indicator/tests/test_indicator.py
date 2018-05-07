#!python
"""
Project: Technical Indicators
Package: indicator
Author: Anuraag Rai Kochhar
Email: arkochhar@hotmail.com
Repository: https://github.com/arkochhar/Technical-Indicators
Version: 1.0.0
License: GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007
"""

import pytest
import unittest
import quandl
import time
import numpy as np
import pandas as pd

from .. import EMA, ATR, SuperTrend, MACD, HA, BBand, RSI, Ichimoku


@pytest.fixture
def df():
    # df = quandl.get("NSE/NIFTY_50", api_key='E8LGujxYzNsiUWYDPbGF', start_date='1997-01-01')
    # df.drop(['Shares Traded', 'Turnover (Rs. Cr)'], inplace=True, axis=1)
    # df.columns = [name.lower() for name in df.columns]
    # df.index.name = df.index.name.lower()
    # df.to_parquet('./indicator/tests/fixtures/nse.prq')
    df = pd.read_parquet('./indicator/tests/fixtures/nse.prq')
    return df


def test_EMA(df, colFrom='close', test_period=5, ignore=0, forATR=False):
    colTo = 'ema_' + str(test_period)

    # Actual function to test compare
    print('EMA Test with alpha {}'.format(forATR))
    start = time.time()
    EMA(df, colFrom, colTo, test_period, alpha=forATR)
    end = time.time()
    print('Time taken by Pandas computations for EMA {}'.format(end - start))

    start = time.time()
    coef = 2 / (test_period + 1)
    periodTotal = 0
    colTest = colTo + '_test'
    df.reset_index(inplace=True)
    for i in range(0, len(df)):
        if (i < ignore):
            df.at[i, colTest] = 0
        elif (i < ignore + test_period - 1):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = 0
        elif (i < ignore + test_period):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = (periodTotal / test_period)
        else:
            if (forATR == True):
                df.at[i, colTest] = (((df.at[i - 1, colTest] * (test_period - 1)) + df.at[i, colFrom]) / test_period)
            else:
                df.at[i, colTest] = (((df.at[i, colFrom] - df.at[i - 1, colTest]) * coef) + df.at[i - 1, colTest])

    df.set_index('date', inplace=True)
    end = time.time()
    print('Time taken by manual computations for EMA {}'.format(end-start))

    df[colTest + '_check'] = df[colTo].round(6) == df[colTest].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[colTest + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[colTest + '_check'].sum() / len(df)) * 100, 2)))


def test_ATR(df, test_period=7):
    print('ATR Test')
    start = time.time()
    ATR(df, 7)
    end = time.time()
    print('Time taken by Pandas computations for ATR {}'.format(end-start))

    ignore=0
    start = time.time()
    periodTotal = 0
    colFrom = 'TR'
    colTo = 'ATR_7'
    colTest = colTo + '_test'

    df['h-l'] = df['high'] - df['low']
    df['h-yc'] = abs(df['high'] - df['close'].shift())
    df['l-yc'] = abs(df['low'] - df['close'].shift())

    df[colFrom] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
    df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    df.reset_index(inplace=True)
    for i in range(0, len(df)):
        if (i < ignore):
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period - 1):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = (periodTotal / test_period)
        else:
            df.at[i, colTest] = (((df.at[i - 1, colTest] * (test_period - 1)) + df.at[i, colFrom]) / test_period)
    df.set_index('date', inplace=True)
    end = time.time()
    print('Time taken by manual computations for ATR {}'.format(end-start))

    df[colTest + '_check'] = df[colTo].round(6) == df[colTest].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[colTest + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[colTest + '_check'].sum() / len(df)) * 100, 2)))


def test_MACD(df):
    print('MACD Test')
    start = time.time()
    MACD(df)
    end = time.time()
    print('Time taken by Pandas computations for MACD {}'.format(end-start))

    df.reset_index(inplace=True)

    fastEMA = 12
    slowEMA = 26
    signal = 9
    fE = "ema_" + str(fastEMA)
    sE = "ema_" + str(slowEMA)
    macd = "macd_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
    sig = "signal_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
    hist = "hist_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)

    fE_test = fE + '_test'
    sE_test = sE + '_test'
    macd_test = macd + '_test'
    sig_test = sig + '_test'
    hist_test = hist + '_test'

    colFrom = 'close'

    start = time.time()
    # Compute fast EMA
    #EMA(df, base, fE, fastEMA)
    periodTotal = 0
    ignore = 0
    test_period = fastEMA
    colTest = fE_test
    coef = 2 / (test_period + 1)
    for i in range(0, len(df)):
        if (i < ignore):
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period - 1):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = (periodTotal / test_period)
        else:
            df.at[i, colTest] = (((df.at[i, colFrom] - df.at[i - 1, colTest]) * coef) + df.at[i - 1, colTest])

    # Compute slow EMA
    #EMA(df, base, sE, slowEMA)
    periodTotal = 0
    ignore = 0
    test_period = slowEMA
    colTest = sE_test
    coef = 2 / (test_period + 1)
    for i in range(0, len(df)):
        if (i < ignore):
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period - 1):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = (periodTotal / test_period)
        else:
            df.at[i, colTest] = (((df.at[i, colFrom] - df.at[i - 1, colTest]) * coef) + df.at[i - 1, colTest])

    # Compute MACD
    df[macd_test] = np.where(np.logical_and(np.logical_not(df[fE_test] == 0), np.logical_not(df[sE_test] == 0)), df[fE_test] - df[sE_test], 0)

    # Compute MACD Signal
    #EMA(df, macd, sig, signal, slowEMA - 1)
    periodTotal = 0
    ignore = 0
    test_period = signal
    colTest = sig_test
    colFrom = macd_test
    coef = 2 / (test_period + 1)
    for i in range(0, len(df)):
        if (i < ignore):
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period - 1):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = 0.00
        elif (i < ignore + test_period):
            periodTotal += df.at[i, colFrom]
            df.at[i, colTest] = (periodTotal / test_period)
        else:
            df.at[i, colTest] = (((df.at[i, colFrom] - df.at[i - 1, colTest]) * coef) + df.at[i - 1, colTest])

    # Compute MACD Histogram
    df[hist_test] = np.where(np.logical_and(np.logical_not(df[macd_test] == 0), np.logical_not(df[sig_test] == 0)), df[macd_test] - df[sig_test], 0)

    end = time.time()
    print('Time taken by manual computations for MACD {}'.format(end-start))

    df.set_index('date', inplace=True)

    print('MACD Stats')
    df[macd + '_check'] = df[macd].round(6) == df[macd_test].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[macd + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[macd + '_check'].sum() / len(df)) * 100, 2)))
    print('Signal Stats')
    df[sig + '_check'] = df[sig].round(6) == df[sig_test].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[sig + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[sig + '_check'].sum() / len(df)) * 100, 2)))
    print('Hist Stats')
    df[hist + '_check'] = df[hist].round(6) == df[hist_test].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[hist + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[hist + '_check'].sum() / len(df)) * 100, 2)))


def test_SuperTrend(df, period=10, multiplier=3):
    atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)
    st_test = st + '_test'
    stx_test = stx + '_test'

    print('SuperTrend Test')
    start = time.time()
    SuperTrend(df, period, multiplier)
    end = time.time()
    print('Time taken by Pandas computations for SuperTrend {}'.format(end-start))

    #start = time.time()
    ATR(df, period)
    df['basic_ub_t'] = (df['high'] + df['low']) / 2 + multiplier * df[atr]
    df['basic_lb_t'] = (df['high'] + df['low']) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df['final_ub_t'] = 0.00
    df['final_lb_t'] = 0.00
    for i in range(period, len(df)):
        df.ix[i, 'final_ub_t'] = df.ix[i, 'basic_ub_t'] if df.ix[i, 'basic_ub_t'] < df.ix[i - 1, 'final_ub_t'] or df.ix[i - 1, 'close'] > df.ix[i - 1, 'final_ub_t'] else df.ix[i - 1, 'final_ub_t']
        df.ix[i, 'final_lb_t'] = df.ix[i, 'basic_lb_t'] if df.ix[i, 'basic_lb_t'] > df.ix[i - 1, 'final_lb_t'] or df.ix[i - 1, 'close'] < df.ix[i - 1, 'final_lb_t'] else df.ix[i - 1, 'final_lb_t']

    # Set the Supertrend value
    df[st_test] = 0.00
    for i in range(period, len(df)):
        df.ix[i, st_test] = df.ix[i, 'final_ub_t'] if df.ix[i - 1, st_test] == df.ix[i - 1, 'final_ub_t'] and df.ix[i, 'close'] <= df.ix[i, 'final_ub_t'] else \
                            df.ix[i, 'final_lb_t'] if df.ix[i - 1, st_test] == df.ix[i - 1, 'final_ub_t'] and df.ix[i, 'close'] >  df.ix[i, 'final_ub_t'] else \
                            df.ix[i, 'final_lb_t'] if df.ix[i - 1, st_test] == df.ix[i - 1, 'final_lb_t'] and df.ix[i, 'close'] >= df.ix[i, 'final_lb_t'] else \
                            df.ix[i, 'final_ub_t'] if df.ix[i - 1, st_test] == df.ix[i - 1, 'final_lb_t'] and df.ix[i, 'close'] <  df.ix[i, 'final_lb_t'] else 0.00

    # index = df.index.name
    # df.reset_index(inplace=True)

    # # Compute final upper and lower bands
    # for i in range(0, len(df)):
        # if i < period:
            # df.set_value(i, 'basic_ub', 0.00)
            # df.set_value(i, 'basic_lb', 0.00)
            # df.set_value(i, 'final_ub', 0.00)
            # df.set_value(i, 'final_lb', 0.00)
        # else:
            # df.set_value(i, 'final_ub', (df.get_value(i, 'basic_ub')
                                         # if df.get_value(i, 'basic_ub') < df.get_value(i-1, 'final_ub') or df.get_value(i-1, 'close') > df.get_value(i-1, 'final_ub')
                                         # else df.get_value(i-1, 'final_ub')))
            # df.set_value(i, 'final_lb', (df.get_value(i, 'basic_lb')
                                         # if df.get_value(i, 'basic_lb') > df.get_value(i-1, 'final_lb') or df.get_value(i-1, 'close') < df.get_value(i-1, 'final_lb')
                                         # else df.get_value(i-1, 'final_lb')))

    # # Set the Supertrend value
    # for i in range(0, len(df)):
        # if i < period:
            # df.set_value(i, st, 0.00)
        # else:
            # df.set_value(i, st, (df.get_value(i, 'final_ub')
                                 # if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, 'close') <= df.get_value(i, 'final_ub')))
                                 # else (df.get_value(i, 'final_lb')
                                       # if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_ub')) and (df.get_value(i, 'close') > df.get_value(i, 'final_ub')))
                                       # else (df.get_value(i, 'final_lb')
                                             # if ((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, 'close') >= df.get_value(i, 'final_lb')))
                                             # else (df.get_value(i, 'final_ub')
                                                   # if((df.get_value(i-1, st) == df.get_value(i-1, 'final_lb')) and (df.get_value(i, 'close') < df.get_value(i, 'final_lb')))
                                                   # else 0.00
                                                  # )
                                            # )
                                      # )
                                # )
                        # )

    # if index:
        # df.set_index(index, inplace=True)

    # Mark the trend direction up/down
    df[stx_test] = np.where((df[st_test] > 0.00), np.where((df['close'] < df[st_test]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub_t', 'basic_lb_t', 'final_ub_t', 'final_lb_t'], inplace=True, axis=1)
    end = time.time()
    print('Time taken by manual computations for SuperTrend {}'.format(end-start))

    print('ST Stats')
    df[st + '_check'] = df[st].round(6) == df[st_test].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[st + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[st + '_check'].sum() / len(df)) * 100, 2)))
    print('STX Stats')
    df[stx + '_check'] = df[stx] == df[stx_test]
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df[stx + '_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df[stx + '_check'].sum() / len(df)) * 100, 2)))

    #print(df[['basic_ub', 'basic_ub_t', 'basic_lb', 'basic_lb_t']])
    #print(df[['final_ub', 'final_ub_t', 'final_lb', 'final_lb_t']])
    #print(df[['ST_7_2', 'STX_7_2', 'ST_7_2_test', 'STX_7_2_test', 'ST_7_2_check', 'STX_7_2_check']])


def test_HA(df):
    print('HA_One Test')
    start = time.time()
    HA(df)
    end = time.time()
    print('Time taken by Pandas computations of HA {}'.format(end-start))

    # Method 1
    start = time.time()
    df['HA_close_t']=(df['open']+ df['high']+ df['low']+df['close'])/4
    for i in range(0, len(df)):
        if i == 0:
            df.ix[i,'HA_open_t'] = (df.ix[i,'open'] + df.ix[i,'open']) / 2
        else:
            df.ix[i,'HA_open_t'] = (df.ix[i - 1,'HA_open_t'] + df.ix[i - 1,'HA_close_t']) / 2

    df['HA_high_t']=df[['HA_open_t','HA_close_t','high']].max(axis=1)
    df['HA_low_t']=df[['HA_open_t','HA_close_t','low']].min(axis=1)
    end = time.time()
    print('Time taken by manual computations method 1 of HA {}'.format(end-start))

    # # Method 2
    # start = time.time()
    # df['HA_close_t2'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    # idx = df.index.name
    # df.reset_index(inplace=True)

    # for i in range(0, len(df)):
        # if i == 0:
            # df.set_value(i, 'HA_open_t2', ((df.get_value(i, 'open') + df.get_value(i, 'close')) / 2))
        # else:
            # df.set_value(i, 'HA_open_t2', ((df.get_value(i - 1, 'HA_open_t2') + df.get_value(i - 1, 'HA_close_t2')) / 2))

    # if idx:
        # df.set_index(idx, inplace=True)

    # df['HA_high_t2']=df[['HA_open_t2', 'HA_close_t2', 'high']].max(axis=1)
    # df['HA_low_t2']=df[['HA_open_t2', 'HA_close_t2', 'low']].min(axis=1)
    # end = time.time()
    # print('Time taken by manual computations method 2 of HA {}'.format(end-start))

    print('open Stats')
    df['HA_open_check'] = df['HA_open'].round(6) == df['HA_open_t'].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df['HA_open_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df['HA_open_check'].sum() / len(df)) * 100, 2)))
    print('high Stats')
    df['HA_high_check'] = df['HA_high'].round(6) == df['HA_high_t'].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df['HA_high_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df['HA_high_check'].sum() / len(df)) * 100, 2)))
    print('low Stats')
    df['HA_low_check'] = df['HA_low'].round(6) == df['HA_low_t'].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df['HA_low_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df['HA_low_check'].sum() / len(df)) * 100, 2)))
    print('close Stats')
    df['HA_close_check'] = df['HA_close'].round(6) == df['HA_close_t'].round(6)
    print('\tTotal Rows: {}'.format(len(df)))
    print('\tColumns Match: {}'.format(df['HA_close_check'].sum()))
    print('\tSuccess Rate: {}%'.format(round((df['HA_close_check'].sum() / len(df)) * 100, 2)))
