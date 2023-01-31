import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

dataFiles = ['XRP.csv','BTC.csv']
nPlots = len(dataFiles)

addVolume = False

for data in dataFiles:
    df = pd.read_csv(data)
    
    df.rename(columns={'open': 'Open', 
                       'close': 'Close', 
                       'high': 'High', 
                       'low': 'Low', 
                       'volume': 'Volume',
                       'dtime': 'DateTime'},
              inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    mpf.plot(df, type='line', style='yahoo', figsize=(14,4))#, type='line' or 'candle')