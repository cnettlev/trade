#!/usr/bin/env python3
# coding: utf-8

# **Requirements**
# 
# pip install krakenex  
# pip install pykrakenapi

from krakenex import API
from pykrakenapi import KrakenAPI
from time import sleep
from datetime import datetime
from os import path
import pandas as pd
from marketList import markets

from retreiveOptions import options

k = KrakenAPI(API())

# # CRYPTO based markets
# markets = ['XRPUSD','BTCUSD', 'ETHUSD']

# # USD based markets
# markets += ['USDCHF', 'USDJPY', 'USDCAD']

# # EUR based markets
# markets += ['EURUSD', 'EURCAD', 'EURJPY', 'EURCHF', 'EURGBP', 'EURAUD']

# # AUD based markets
# markets += ['AUDJPY', 'AUDUSD']

# # USDT based markets
# markets += ['USDTCAD','USDTEUR','USDTGBP','USDTJPY','USDTCHF','USDTAUD']

# # USDC based markets
# markets += ['USDCUSD','USDCEUR','USDCUSDT','USDCAUD','USDCGBP']

# # Other markets
# markets += ['GBPUSD']

getSince = { 'default': datetime(2020,1,1).timestamp() }

for market in markets:
    marketFile = options.folder+market + '.csv'
    print('Retreiving last record from'+marketFile+'...')
    if path.exists(marketFile):

        # import os
        # with open(marketFile, 'rb') as f:
        #     try:  # catch OSError in case of a one line file 
        #         f.seek(-2, os.SEEK_END)
        #         while f.read(1) != b'\n':
        #             f.seek(-2, os.SEEK_CUR)
        #     except OSError:
        #         f.seek(0)
        #     last_line = f.readline().decode()
# 
        # print(last_line)

        df = pd.read_csv(marketFile)
        getSince[market] = df.iloc[-1].time
    else:
        getSince[market] = getSince['default']

# exit(0)
        
print(getSince)


# Get open high low close data since date

# interval = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600} [min]

def extendCSV(df,fileName):
    if path.exists(fileName) and options.forceNewLine:
        with open(fileName, 'a') as f:
            f.write('\n')    
    df.to_csv(fileName, mode='a', header=not path.exists(fileName))

while True:    
    for market in markets:
        print('Retreiving '+market+' at',datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        print('\t last record: \t',datetime.utcfromtimestamp(getSince[market]))
    
        ohlc, getSince[market] = k.get_ohlc_data(pair = market,interval = 1,since = getSince[market],ascending = True)
        
        print('\t first retreived: \t',datetime.utcfromtimestamp(ohlc.iloc[0].time))
        print('\t last retreived: \t',datetime.utcfromtimestamp(getSince[market]))
    
        extendCSV(ohlc.iloc[:-1], options.folder+market+'.csv')
        sleep(1)

    if options.singleRetreive:
        exit(0)
    
    print(' ... waiting 0', end = "")
    for i in range(6):
        sleep(600)
        print(',', 10*(i+1), end = "")
    print('. Done! let\'s go again')

