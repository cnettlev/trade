#!/usr/bin/env python
# coding: utf-8

# To do:
# 
# - Fix RSI
# - Schaff Trend Cycle (STC)
#   - Similar to MACD but takes into consideration a cycle component.
# - Bollinger bands
#   - If the price moves outside the upper parameters it could be overbought. Moving below the lower band means oversold.
# - Fibonacci retracement
# - Ichimoku cloud
# - Standard deviation

import pandas as pd
import numpy as np

def sampleData(folder = 'market-data/'):
    from marketList import markets
    return pd.read_csv(folder+markets[0]+'.csv')


def computeLostData(df, dtime=60):
    lackOfData = np.where(df.time.diff()!=dtime)[0] # More than a minute between datapoints
    df['lost-data'] = 0
    df.loc[lackOfData,'lost-data'] = 1
    
    return df

def requiresLostData(df):
    if not 'lost-data' in df.columns:
        return computeLostData(df)
    else:
        return df

def computeOBV(df):
    df = requiresCloseDiff(df)
    df = requiresLostData(df)
    
    df['OBV'] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    signedVolume = df['volume'] * np.sign(df.close.diff())
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi:lDim-1,'OBV'] = signedVolume.loc[lDi:lDim].cumsum(axis = 0, skipna = True)
    df.loc[lackOfData[-1]:,'OBV'] = signedVolume.loc[lackOfData[-1]:].cumsum(axis = 0, skipna = True)
    
    df['OBV-sign'] = np.sign(df['OBV'])
    df['OBV-price-divergence'] = (df['dclose-sign']*df['OBV-sign']) == -1
    # df['OBV-price-divergence'] = (np.sign(df.close.diff())*df['OBV-sign']) == -1
    df.loc[df['lost-data'] == 1,'OBV-price-divergence'] = False
    
    return df

def EWM(df,key,span=14):
    sma = df[key].rolling(window=span, min_periods=span).mean()[:span]
    rest = df[key][span:]
    return pd.concat([sma, rest]).ewm(span=span, adjust=False).mean()

def computeEWM(df,key,span=14):
    df = requiresLostData(df)    
    newKey = 'e'+key
    df[newKey] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi+1:lDim-1,newKey] = EWM(df.loc[lDi+1:lDim-1],key)
    df.loc[lackOfData[-1]:,newKey] = EWM(df.loc[lackOfData[-1]:],key)
    
    return df

def roll(df,key,span=14):
    df = requiresLostData(df)    
    newKey = 'A'+key
    df[newKey] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi+1:lDim-1,newKey] = df.loc[lDi+1:lDim-1,key].rolling(window=span,min_periods=span).mean()
    df.loc[lackOfData[-1]:,newKey] = df.loc[lackOfData[-1]:,key].rolling(window=span,min_periods=span).mean()
    
    return df

def divideWithZero(df,num,div,outKey,whenZero=1):
    df[outKey] = whenZero
    
    df.loc[div != 0,outKey] = num.loc[div != 0]/div.loc[div != 0]
    
    return df

def computeATR(df, span = 14):
    df = requiresLostData(df)
    
    df['TR'] = 0
    df['ATR'] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        maxVal = np.max([df['high'][lDi+1:lDim],df['close'][lDi:lDim-1]],axis=0)
        minVal = np.min([df['low'][lDi+1:lDim],df['close'][lDi:lDim-1]],axis=0)
        df.loc[lDi+1:lDim-1,'TR'] = maxVal-minVal
        
    df.loc[lackOfData[-1]+1:,'TR'] = np.max([df['high'][lackOfData[-1]+1:],df['close'][lackOfData[-1]:-1]],axis=0) -                                             np.min([df['low'][lackOfData[-1]+1:],df['close'][lackOfData[-1]:-1]],axis=0)
            
    df = roll(df,'TR') # Creates ATR
    
    df = computeEWM(df,key='ATR',span=span)
    
    return df


def requiresATR(df):
    if not 'ATR' in df.columns:
        return computeATR(df)
    else:
        return df

def computeAD(df):
    df = requiresLostData(df)
    df = requiresCloseDiff(df)
    df['AD'] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    df = divideWithZero(df,
                        num=(df['close'] - df['low'])-(df['high'] - df['close']),
                        div=df['high']-df['low'],
                        outKey='moneyFlow')
    
    moneyFlowTimesVolume = df['volume'] * df['moneyFlow']
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi:lDim-1,'AD'] = moneyFlowTimesVolume.loc[lDi:lDim].cumsum(axis = 0, skipna = True)
        
    df.loc[lackOfData[-1]:,'AD'] = moneyFlowTimesVolume.loc[lackOfData[-1]:].cumsum(axis = 0, skipna = True)
    
    
    df['AD-sign'] = np.sign(df['AD'])
    df['AD-price-divergence'] = (df['dclose-sign']*df['AD-sign']) == -1
    df.loc[df['lost-data'] == 1,'AD-price-divergence'] = False
    
    return df    

def computeDiff(df,key,span=14):
    df = requiresLostData(df)
    newKey = 'd'+key
    
    df[newKey] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi:lDim-1,newKey] = df[key].loc[lDi:lDim].diff()    
    df.loc[lackOfData[-1]:,newKey] = df[key].loc[lackOfData[-1]:].diff()
    
    df = computeEWM(df,key=newKey,span=span) # creates 'e'+newKey
    
    df[newKey+'-sign'] = np.sign(df[newKey])
    
    return df

##### 

def computeHighDiff(df):
    df = requiresLostData(df)
    
    return computeDiff(df,'high')

def requiresHighDiff(df):
    if not 'dhigh' in df.columns:
        return computeHighDiff(df)
    else:
        return df
    
def computeLowDiff(df):
    df = requiresLostData(df)
    
    return computeDiff(df,'low')

def requiresLowDiff(df):
    if not 'dlow' in df.columns:
        return computeLowDiff(df)
    else:
        return df
    
def computeCloseDiff(df):
    df = requiresLostData(df)
    
    return computeDiff(df,'close')

def requiresCloseDiff(df):
    if not 'dclose' in df.columns:
        return computeCloseDiff(df)
    else:
        return df
    
def computeOpenDiff(df):
    df = requiresLostData(df)
    
    return computeDiff(df,'open')

def requiresOpenDiff(df):
    if not 'dopen' in df.columns:
        return computeOpenDiff(df)
    else:
        return df

def computeADX(df):
    df = requiresHighDiff(df)
    df = requiresLowDiff(df)
    df = requiresATR(df)
    
    df['+DI'] = 100.0 * (df['edhigh']/df['ATR'])
    df['-DI'] = 100.0 * (df['edlow']/df['ATR'])
    df['DX']  = 100.0 * (np.abs((df['+DI']-df['-DI'])/(df['+DI']+df['-DI'])))
    
    df = roll(df,key='DX') # Creates ADX
    
    df['ADI']  = 0
    df.loc[(df['ADX']<=40) & (df['ADX']>20) ,'ADI'] = 1
    df.loc[df['ADX']>40,'ADI'] = 3
    
    
    df['ADI2']  = 0
    df.loc[(df['-DI']<df['+DI']) & (df['ADX']>20) ,'ADI2'] = 1
    df.loc[(df['-DI']>df['+DI']) & (df['ADX']>20) ,'ADI2'] = -1
    
    return df

def computeAroon(df):
    df = requiresCloseDiff(df)
    df = requiresLostData(df)

    dprice = pd.DataFrame({'up': df['close'] == 1,'down': df['dclose-sign'] == -1})
    dprice['lost-data'] = df['lost-data']
    dprice = roll(dprice,'up',span=25)
    dprice = roll(dprice,'down',span=25)
    df['aroon-up']    = 100.0*dprice['Aup']
    df['aroon-down']  = 100.0*dprice['Adown']
    
    return df


def computeMACD(df):
    df = requiresLostData(df)   
    
    eprice = pd.DataFrame({'wm12': df['close'],'wm26': df['close']})
    eprice['lost-data'] = df['lost-data']
    eprice = computeEWM(eprice,key='wm12',span=12) #creates ewm12
    eprice = computeEWM(eprice,key='wm26',span=26) #creates ewm26
    eprice['wm9'] = eprice['ewm12'] - eprice['ewm26']
    eprice = computeEWM(eprice,key='wm9',span=9) #creates ewm9
    
    df['MACD'] = np.sign(eprice['wm9']-eprice['ewm9'])
    df['MACD-price-divergence'] = (df['dclose-sign']*df['MACD']) == -1
    df.loc[df['lost-data'] == 1,'MACD-price-divergence'] = False
    
    return df


def requiresMACD(df):
    if not 'MACD' in df.columns:
        return computeMACD(df)
    else:
        return df


def computeRSI(df, span=14):
    df = requiresCloseDiff(df)
    df = requiresLostData(df)
    
    df['RSI'] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    
    gainLoss = pd.DataFrame({'gain': df['dclose'],'loss': df['dclose']})
    gainLoss.loc[gainLoss['gain']<=0,'gain'] = 0
    gainLoss.loc[gainLoss['loss']>=0,'loss'] = 0
    
    gainLoss['lost-data'] = df['lost-data']
    gainLoss = computeEWM(gainLoss,key='gain',span=12) #creates egain
    gainLoss = computeEWM(gainLoss,key='loss',span=26) #creates eloss
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        if lDi+span < lDim:
            df.loc[lDi:lDi+span-1,'RSI'] = 100.0 - 100.0/(1.0+df.loc[df['dclose-sign']==1,'dclose'].mean()/df.loc[df['dclose-sign']==-1,'dclose'].mean())
            df.loc[lDi+span:lDim,'RSI'] = 100.0 - 100.0/(1.0+gainLoss.loc[lDi+span:lDim,'egain']/gainLoss.loc[lDi+span:lDim,'eloss'])
        else:
            df.loc[lDi:lDim-1,'RSI'] = 100.0 - 100.0/(1.0+df.loc[df['dclose-sign']==1,'dclose'].mean()/df.loc[df['dclose-sign']==-1,'dclose'].mean())
    
    df.loc[lDim:,'RSI'] = 100.0 - 100.0/(1.0+gainLoss.loc[lDim:,'egain']/gainLoss.loc[lDim:,'eloss'])
    
    df['RSI-signal'] = 0
    df.loc[df['RSI']>70,'RSI-signal'] = 1
    df.loc[df['RSI']<30,'RSI-signal'] = -1
    
    df = computeEWM(df,key='RSI-signal')
    
    df['RSI-price-divergence'] = (df['dclose-sign']*np.sign(df['RSI'])) == -1
    df.loc[df['lost-data'] == 1,'RSI-price-divergence'] = False
            
    return df


def computeSO(df,keys=['high','low','close'],span=14,oKey='SO',computeSignal=True):
    df = requiresLostData(df)
    
    highLow = pd.DataFrame({'high': df[keys[0]],'low': df[keys[1]]})
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        highLow.loc[lDi:lDim-1,'high'] = highLow.loc[lDi:lDim-1,'high'].rolling(span,min_periods=span).max()
        highLow.loc[lDi:lDim-1,'low'] = highLow.loc[lDi:lDim-1,'low'].rolling(span,min_periods=span).min()
    
    highLow.loc[lDim:,'high'] = highLow['high'].rolling(span,min_periods=span).max()    
    highLow.loc[lDim:,'low'] = highLow['low'].rolling(span,min_periods=span).min()
    
    df = divideWithZero(df,
                        num=df[keys[2]] - highLow['low'],
                        div=highLow['high'] - highLow['low'],
                        outKey=oKey)
    
    df[oKey] = 100.0 * df[oKey]
    
    if computeSignal:
        signalKey = oKey+'-signal'
        df[signalKey] = 0
        df.loc[df[oKey]>80,signalKey] = 1
        df.loc[df[oKey]<20,signalKey] = -1
    
        df = computeEWM(df,key=signalKey)
    
    return df

def computeSTC(df):
    df = requiresLostData(df)
    
    eprice = pd.DataFrame({'wm23': df['close'],'wm50': df['close']})
    eprice['lost-data'] = df['lost-data']
    eprice = computeEWM(eprice,key='wm23',span=23) #creates ewm23
    eprice = computeEWM(eprice,key='wm50',span=50) #creates ewm50
    
    eprice['MACD'] = eprice['ewm23'] - eprice['ewm50']
    eprice = computeSO(eprice,keys=['MACD','MACD','MACD'],span=10,oKey='KMACD',computeSignal=False)
    eprice = computeEWM(eprice,key='KMACD',span=3) #creates eKMACD
    
    
    df = divideWithZero(df,
                        num=eprice['MACD']-eprice['KMACD'],
                        div=eprice['eKMACD']-eprice['KMACD'],
                        outKey='STC')
    df['STC'] = 100.0 * df['STC']
    
    signalKey = 'STC-signal'
    df[signalKey] = 0
    df.loc[df['STC']>=75,signalKey] = -1
    df.loc[df['STC']<=25,signalKey] = 1

    df = computeEWM(df,key=signalKey)
    
    return df

def computeBB(df,nSD=2,span=20): # Borllinger bands
    df = requiresLostData(df)
    
    tprice = pd.DataFrame({'typicalPrice': (df['high']+df['low']+df['close'])/3.0})
    
    tprice['lost-data'] = df['lost-data']
    tprice = roll(tprice,key='typicalPrice',span=span) # creates AtypicalPrice
    tprice['std'] = 0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        tprice.loc[lDi:lDim-1,'std'] = tprice.loc[lDi:lDim-1,'typicalPrice'].rolling(span,min_periods=span).std()
    
    tprice.loc[lDim:,'std'] = tprice['typicalPrice'].rolling(span,min_periods=span).std() 
    
    df['BOLU'] = tprice['AtypicalPrice'] + nSD * tprice['std']
    df['BOLD'] = tprice['AtypicalPrice'] - nSD * tprice['std']

    df['BB'] = 0
    
    df.loc[(df['close'] > df['BOLU']) | (df['close'] < df['BOLD']),'BB'] = 1
    
    return df



