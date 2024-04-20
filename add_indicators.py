#!/usr/bin/env python
# coding: utf-8

# To do:
# 
# - Check the Schaff Trend Cycle (STC) timeframes
#   - Similar to MACD but takes into consideration a cycle component.
# - Bollinger bands
#   - If the price moves outside the upper parameters it could be overbought. Moving below the lower band means oversold.
# - Fibonacci retracement
# - Ichimoku cloud
# - Standard deviation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def loadData(folder: str = 'market-data/', key: str = 'XRPUSD' ) -> pd.DataFrame:
    return pd.read_csv(folder+key+'.csv')

def sampleData(folder: str = 'market-data/', dataIndex: int = 0 ) -> pd.DataFrame:
    from marketList import markets
    return pd.read_csv(folder+markets[dataIndex]+'.csv')


def computeLostData(df: pd.DataFrame, dtime: int = 60):
    """
    Computes the lost data in a DataFrame by adding a column 'lost-data' that is 1 if
    there is more than a minute between datapoints.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    dtime (int): The time interval in seconds to consider as a threshold for lost data. Default is 60 seconds.
    """
    df['lost-data'] = 0
    lackOfData = np.where(df.time.diff()!=dtime)[0] # More than a minute between datapoints
    df['lost-data'] = 0
    df.loc[lackOfData,'lost-data'] = 1

def requiresLostData(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the DataFrame contains the 'lost-data' column.
    If not, computes the lost data on the received dataframe.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """
    if not 'lost-data' in df.columns:
        computeLostData(df)

def OBV(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the On Balance Volume (OBV) for an entire dataframe.

    Parameters:
    df (pd.DataFrame): Dataframe with columns 'close' and 'volume'.

    Returns:
    np.ndarray: Array containing the On Balance Volume.
    """

    return (df['volume'] * df['dclose-sign']).cumsum()
    
    # return np.where(df['close'] > df['close'].shift(1), df['volume'],
    #                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()

def computeOBV(df: pd.DataFrame):
    """
    Computes the On Balance Volume (OBV) indicator for a given DataFrame
    through sections between lost data using OBV function.

    Parameters:
    - df: DataFrame
        The input DataFrame containing the necessary columns for calculation.
    """
    requiresLostData(df)
    requiresCloseDiff(df)

    df['OBV'] = 0.0
    lackOfData = df.loc[df['lost-data'] == 1].index

    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi:lDim-1,'OBV'] = OBV(df.loc[lDi:lDim-1])
    df.loc[lackOfData[-1]:,'OBV'] = OBV(df.loc[lackOfData[-1]:])

    df['OBV-sign'] = np.sign(df['OBV'])
    df['OBV-price-divergence'] = (df['dclose-sign']*df['OBV-sign']) == -1
    df.loc[df['lost-data'] == 1,'OBV-price-divergence'] = False


    # self.roll(self.data,'OBV',span=M)
    # self.data['OBV-price-divergence'] = (np.sign(self.data['dclose-sign']*self.data['OBV-sign']) == -1)
    # self.data.loc[self.data['lost-data'] == 1,'OBV-price-divergence'] = False


# def previousversion_computeOBV(df):
    # df['OBV'] = 0
    
    # lackOfData = df.loc[df['lost-data'] == 1].index
        
    # signedVolume = df['volume'] * np.sign(df.close.diff())
        
    # for lDi,lDim in zip(lackOfData,lackOfData[1:]):
    #     df.loc[lDi:lDim-1,'OBV'] = signedVolume.loc[lDi:lDim].cumsum(axis = 0, skipna = True)
    # df.loc[lackOfData[-1]:,'OBV'] = signedVolume.loc[lackOfData[-1]:].cumsum(axis = 0, skipna = True)
    
    # df['OBV-sign'] = np.sign(df['OBV'])
    # df['OBV-price-divergence'] = (df['dclose-sign']*df['OBV-sign']) == -1
    # # df['OBV-price-divergence'] = (np.sign(df.close.diff())*df['OBV-sign']) == -1
    # df.loc[df['lost-data'] == 1,'OBV-price-divergence'] = False
    
    # return df

def EWM(df: pd.DataFrame,key: str ,span: int = 14) -> pd.Series:
    """
    Calculates the Exponential Weighted Moving Average (EWMA) for a given DataFrame column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - key (str): The column name to calculate the EWMA for.
    - span (int): The span or window size for the EWMA calculation. Default is 14.

    Returns:
    - Series: The EWMA values for the specified column.
    """
    sma = df[key].rolling(window=span, min_periods=1).mean()[:span]
    rest = df[key][span:]
    return pd.concat([sma, rest]).ewm(span=span, adjust=False, min_periods = 1).mean()

def computeEWM(df: pd.DataFrame,key: str,span: int = 14, fillNaN: bool = True):
    """
    Computes the Exponential Weighted Moving Average (EWMA) for a given DataFrame
    through sections between lost data using EWM function.

    Parameters:
    - df (DataFrame): The input DataFrame containing the necessary columns for calculation.
    - key (str): The column name to calculate the EWMA for.
    - span (int): The span or window size for the EWMA calculation. Default is 14.
    - fillNaN (bool): Whether to fill NaN values with original values. Default is True.
    """
    requiresLostData(df)    
    newKey = 'e'+key
    df[newKey] = 0.0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi+1:lDim-1,newKey] = EWM(df.loc[lDi+1:lDim-1],key,span)
    df.loc[lackOfData[-1]:,newKey] = EWM(df.loc[lackOfData[-1]:],key,span)

    if fillNaN:
        df.fillna({newKey:df[key]},inplace=True)

def roll(df: pd.DataFrame,key: str,span: int = 14, fillNaN: bool = True):
    """
    Applies rolling mean calculation to a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to perform the rolling mean calculation on.
        key (str): The column name to apply the rolling mean calculation to.
        span (int, optional): The window size for the rolling mean calculation. Defaults to 14.
        fillNaN (bool, optional): Whether to fill NaN values with original values. Defaults to True.
    """
    requiresLostData(df)    
    newKey = 'A'+key
    df[newKey] = 0.0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi+1:lDim-1,newKey] = df.loc[lDi+1:lDim-1,key].rolling(window=span,min_periods=span).mean()
    df.loc[lackOfData[-1]:,newKey] = df.loc[lackOfData[-1]:,key].rolling(window=span,min_periods=span).mean()

    if fillNaN:
        df.fillna({newKey:df[key]},inplace=True)

def rollArray(arr: np.ndarray, span: int = 14, mode: str = 'same') -> np.ndarray:
    """
    Applies rolling mean calculation to a given array.

    Args:
        arr (np.ndarray): The input array to perform the rolling mean calculation on.
        span (int, optional): The window size for the rolling mean calculation. Defaults to 14.
        mode (str, optional): The mode parameter for the rolling mean calculation. Defaults to 'full'.
    Returns:
        np.ndarray: The array with the rolling mean values.
    """
    return np.convolve(arr, np.ones(span), mode = mode) / span

def divideWithZero(df: pd.DataFrame,num: str,div: str,outKey: str,whenZero: float = 1.0) -> pd.DataFrame:
    """
    Divides the values in the 'num' column of the DataFrame 'df' by the values in the 'div' column,
    and stores the result in the 'outKey' column. If the divisor is zero, the 'whenZero' value is used instead.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        num (str): The name of the column containing the numerator values.
        div (str): The name of the column containing the divisor values.
        outKey (str): The name of the column to store the result.
        whenZero (float, optional): The value to use when the divisor is zero. Defaults to 1.0.

    """
    df[outKey] = whenZero
    
    df.loc[div != 0,outKey] = num.loc[div != 0]/div.loc[div != 0]
    

def computeATR_prev(df: pd.DataFrame, span: int = 14):
    """
    Computes the Average True Range (ATR) indicator for a given DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the necessary columns ('high', 'low', 'close', 'lost-data').
        span (int): The span parameter for computing the Exponential Weighted Moving Average (EWM) of ATR. Default is 14.
    """
    requiresLostData(df)

    df['TR'] = 0
    df['ATR'] = 0

    lackOfData = df.loc[df['lost-data'] == 1].index

    for lDi, lDim in zip(lackOfData, lackOfData[1:]):
        maxVal = np.max([df['high'][lDi + 1:lDim], df['close'][lDi:lDim - 1]], axis=0)
        minVal = np.min([df['low'][lDi + 1:lDim], df['close'][lDi:lDim - 1]], axis=0)
        df.loc[lDi + 1:lDim - 1, 'TR'] = maxVal - minVal

    df.loc[lackOfData[-1] + 1:, 'TR'] = np.max([df['high'][lackOfData[-1] + 1:], df['close'][lackOfData[-1]:-1]], axis=0) - \
                                       np.min([df['low'][lackOfData[-1] + 1:], df['close'][lackOfData[-1]:-1]], axis=0)

    roll(df, 'TR')  # Creates ATR

    computeEWM(df, key='ATR', span=span)

def computeATR(df: pd.DataFrame, span: int = 14):
    """
    Computes the Average True Range (ATR) indicator for a given DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the necessary columns ('high', 'low', 'close', 'lost-data').
        span (int): The span parameter for computing the Exponential Weighted Moving Average (EWM) of ATR. Default is 14.
    """
    requiresLostData(df)

    df['TR'] = 0.0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        high_low = df['high'][lDi+1:lDim + 1].values-df['low'][lDi+1:lDim + 1].values
        high_close = np.abs(df['high'][lDi+1:lDim + 1].values-df['close'][lDi:lDim].values)
        low_close = np.abs(df['low'][lDi+1:lDim + 1].values-df['close'][lDi:lDim].values)
        df.loc[lDi+1:lDim,'TR'] = np.maximum.reduce([high_low, high_close, low_close])

    high_low = df['high'][lackOfData[-1] + 1:].values-df['low'][lackOfData[-1] + 1:].values
    high_close = np.abs(df['high'][lackOfData[-1] + 1:].values-df['close'][lackOfData[-1]:-1].values)
    low_close = np.abs(df['low'][lackOfData[-1] + 1:].values-df['close'][lackOfData[-1]:-1].values)
    df.loc[lackOfData[-1] + 1:,'TR'] =  np.maximum.reduce([high_low, high_close, low_close])

    roll(df,'TR',span=span) # Creates ATR

def requiresATR(df: pd.DataFrame):
    """
    Checks if the DataFrame has the 'ATR' column.
    If not, it computes the Average True Range (ATR) and adds it to the DataFrame.
    
    Parameters:
    - df: DataFrame
        The input DataFrame.
    """
    if not 'ATR' in df.columns:
        computeATR(df)

def computeAD(df: pd.DataFrame):
    """
    Compute Accumulation/Distribution (AD) indicator for a given DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing necessary columns.

    """
    requiresLostData(df)
    requiresCloseDiff(df)
    df['AD'] = 0.0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    divideWithZero(df,
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

def computeDiff(df: pd.DataFrame, key: str, span: int = 14):
    """
    Computes the difference between consecutive values of a specified column in a DataFrame.
    It also computes the exponential weighted moving average (EWM) and the sign of the difference,
    included as columns ed'key' and ed'key'-sign, respectively.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        key (str): The column name for which to compute the difference.
        span (int, optional): The span parameter for the exponential weighted moving average (EWM). Default is 14.
    """
    requiresLostData(df)
    newKey = 'd'+key
    
    df[newKey] = 0.0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi:lDim-1,newKey] = df[key].loc[lDi:lDim].diff()    
    df.loc[lackOfData[-1]:,newKey] = df[key].loc[lackOfData[-1]:].diff()
    
    computeEWM(df,key=newKey,span=span) # creates 'e'+newKey
    
    df[newKey+'-sign'] = np.sign(df[newKey])
    

def computeHighDiff(df: pd.DataFrame):
    """
    Computes the difference between the high values in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the high values.
    """
    requiresLostData(df)
    computeDiff(df,'high')

def requiresHighDiff(df: pd.DataFrame):
    """
    Checks if the DataFrame contains the 'dhigh' column.
    If not, computes the high difference and returns the updated DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    if not 'dhigh' in df.columns:
        computeHighDiff(df)
    
def computeLowDiff(df: pd.DataFrame):
    """
    Compute the difference between the low values in the DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    requiresLostData(df)
    computeDiff(df,'low')

def requiresLowDiff(df: pd.DataFrame):
    """
    Checks if the 'dlow' column is present in the DataFrame.
    If not, it computes the low difference and adds it as a new column.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if not 'dlow' in df.columns:
        return computeLowDiff(df)
    else:
        return df
    
def computeCloseDiff(df: pd.DataFrame):
    """
    Computes the difference between consecutive close prices in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the close prices.

    Returns:
        pandas.DataFrame: The DataFrame with an additional column representing the difference between consecutive close prices.
    """
    requiresLostData(df)
    computeDiff(df,'close')

def requiresCloseDiff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the 'dclose' column is present in the DataFrame.
    If not, computes the difference between consecutive 'close' values and adds it as a new column.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if not 'dclose' in df.columns:
        computeCloseDiff(df)
    
def computeOpenDiff(df: pd.DataFrame):
    """
    Computes the difference between the open price of each row and the previous row in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the open prices.
    """
    requiresLostData(df)
    computeDiff(df,'open')

def requiresOpenDiff(df: pd.DataFrame):
    """
    Checks if the given DataFrame contains the 'dopen' column.
    If not, it computes the open difference and returns the updated DataFrame.
    If the 'dopen' column already exists, it returns the DataFrame as is.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    if not 'dopen' in df.columns:
        computeOpenDiff(df)

def computeADX_prev(df: pd.DataFrame):
    """
    Compute the Average Directional Index (ADX) and related indicators for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.
    """
    requiresHighDiff(df)
    requiresLowDiff(df)
    requiresATR(df)
    
    df['+DI'] = 100.0 * (df['edhigh']/df['ATR'])
    df['-DI'] = 100.0 * (df['edlow']/df['ATR'])
    df['DX']  = 100.0 * (np.abs((df['+DI']-df['-DI'])/(df['+DI']+df['-DI'])))
    
    roll(df,key='DX') # Creates ADX
    
    df['ADI']  = 0
    df.loc[(df['ADX']<=40) & (df['ADX']>20) ,'ADI'] = 1
    df.loc[df['ADX']>40,'ADI'] = 3
    
    
    df['ADI2']  = 0
    df.loc[(df['-DI']<df['+DI']) & (df['ADX']>20) ,'ADI2'] = 1
    df.loc[(df['-DI']>df['+DI']) & (df['ADX']>20) ,'ADI2'] = -1
    


def computeADX(df: pd.DataFrame):
    """
    Compute the Average Directional Index (ADX) and related indicators for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.
    """
    requiresHighDiff(df)
    requiresLowDiff(df)
    requiresATR(df)

    iLow = -df['dlow']
    dm_plus = np.where((df['dhigh'] > iLow) & (df['dhigh'] > 0), df['dhigh'], 0)
    dm_minus = np.where((iLow > df['dhigh']) & (iLow > 0), iLow, 0)
    
    sdm_plus = rollArray(dm_plus,span=14)
    sdm_minus = rollArray(dm_minus,span=14)

    df['+DI'] = 100.0 * sdm_plus/(df['ATR']+1e-10)
    df['-DI'] = 100.0 * sdm_minus/(df['ATR']+1e-10)
    df['DX']  = 100.0 * (np.abs((df['+DI']-df['-DI'])/(df['+DI']+df['-DI']+1e-10)))
    roll(df,'DX',span=14)
    
    df['ADI']  = 0
    df.loc[(df['ADX']<=40) & (df['ADX']>20) ,'ADI'] = 1
    df.loc[df['ADX']>40,'ADI'] = 3
    
    df['ADI2']  = 0
    df.loc[(df['-DI']<df['+DI']) & (df['ADX']>20) ,'ADI2'] = 1
    df.loc[(df['-DI']>df['+DI']) & (df['ADX']>20) ,'ADI2'] = -1

def computeAroon(df: pd.DataFrame, span: int = 25):
    """
    Compute Aroon indicators for a given DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing necessary columns.
    """
    requiresCloseDiff(df)
    requiresLostData(df)

    # dprice = pd.DataFrame({'up': df['dclose-sign'] == 1,'down': df['dclose-sign'] == -1})
    # dprice['lost-data'] = df['lost-data']
    # roll(dprice,'up',span=25)
    # roll(dprice,'down',span=25)
    # df['aroon-up']    = 100.0*dprice['Aup']
    # df['aroon-down']  = 100.0*dprice['Adown']

    df['aroon-up'] = df['high'].rolling(span, min_periods = 1).apply(lambda x: (np.argmax(x[::-1]) + 1) if np.any(x) else np.nan, raw = True)
    df['aroon-down'] = df['low'].rolling(span, min_periods = 1).apply(lambda x: (np.argmin(x[::-1]) + 1) if np.any(x) else np.nan, raw = True)    

    df['aroon-up'] = 100.0 - 100.0 * df['aroon-up'] / span
    df['aroon-down'] = 100.0 - 100.0 * df['aroon-down'] / span

    # roll(df,'aroon-up',span=span)
    # roll(df,'aroon-down',span=span)

def computeMACD(df: pd.DataFrame):
    """
    Computes the Moving Average Convergence Divergence (MACD) indicator for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.
    """
    requiresLostData(df)   
    requiresCloseDiff(df)
    
    eprice = pd.DataFrame({'wm12': df['close'],'wm26': df['close']})
    eprice['lost-data'] = df['lost-data']
    computeEWM(eprice,key='wm12',span=12) #creates ewm12
    computeEWM(eprice,key='wm26',span=26) #creates ewm26
    eprice['wm9'] = eprice['ewm12'] - eprice['ewm26']
    computeEWM(eprice,key='wm9',span=9) #creates ewm9
    
    df['MACD'] = np.sign(eprice['wm9']-eprice['ewm9'])
    df['MACD-price-divergence'] = (df['dclose-sign']*df['MACD']) == -1
    df.loc[df['lost-data'] == 1,'MACD-price-divergence'] = False
    


def requiresMACD(df: pd.DataFrame):
    """
    Checks if the DataFrame `df` contains the 'MACD' column.
    If not, computes the MACD and adds it to the DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
    """
    if not 'MACD' in df.columns:
        computeMACD(df)


def computeRSI(df: pd.DataFrame, span: int = 14):
    """
    Computes the Relative Strength Index (RSI) for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the necessary columns.
        span (int, optional): The time span used to calculate the RSI. Defaults to 14.
    """
    requiresCloseDiff(df)
    requiresLostData(df)
    
    df['RSI'] = 0.0
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    
    gainLoss = pd.DataFrame({'gain': df['dclose'],'loss': -df['dclose']})
    gainLoss.loc[gainLoss['gain']<=0,'gain'] = 0
    gainLoss.loc[gainLoss['loss']<=0,'loss'] = 0
    
    gainLoss['lost-data'] = df['lost-data']
    computeEWM(gainLoss,key='gain',span=span) #creates egain
    computeEWM(gainLoss,key='loss',span=span) #creates eloss
        
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        df.loc[lDi:lDim,'RSI'] = 100.0 - 100.0/(1.0+gainLoss.loc[lDi:lDim,'egain']/(gainLoss.loc[lDi:lDim,'eloss']+1e-10))
        # if lDi+span < lDim:
        #     df.loc[lDi:lDi+span-1,'RSI'] = 100.0 - 100.0/(1.0-df.loc[df['dclose-sign']==1,'dclose'].mean()/df.loc[df['dclose-sign']==-1,'dclose'].mean())
        #     df.loc[lDi+span:lDim,'RSI'] = 100.0 - 100.0/(1.0+gainLoss.loc[lDi+span:lDim,'egain']/(gainLoss.loc[lDi+span:lDim,'eloss']+1e-10))
        # else:
        #     df.loc[lDi:lDim-1,'RSI'] = 100.0 - 100.0/(1.0-df.loc[df['dclose-sign']==1,'dclose'].mean()/df.loc[df['dclose-sign']==-1,'dclose'].mean())
    
    df.loc[lDim:,'RSI'] = 100.0 - 100.0/(1.0+gainLoss.loc[lDim:,'egain']/(gainLoss.loc[lDim:,'eloss']+1e-10))
    
    df['RSI-signal'] = 0
    df.loc[df['RSI']>70,'RSI-signal'] = 1
    df.loc[df['RSI']<30,'RSI-signal'] = -1
    
    computeEWM(df,key='RSI-signal')
    
    df['RSI-price-divergence'] = (df['dclose-sign']*np.sign(df['RSI'])) == -1
    df.loc[df['lost-data'] == 1,'RSI-price-divergence'] = False

def computeSO(df: pd.DataFrame, span: int = 14, computeOver = ['high','low','close'],
               oKey: str = 'SO', computeSignal: bool = True):
    """
    Compute the Stochastic Oscillator (SO) indicator for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - span (int): The number of periods to consider for calculating the rolling maximum and minimum values. Default is 14.
    - computeOver (list): The list of columns to compute the SO over. Default is ['high','low','close'].
    - oKey (str): The output key for storing the computed SO values in the DataFrame. Default is 'SO'.
    - computeSignal (bool): Whether to compute the signal values based on the SO values. Default is True.
    """
    requiresLostData(df)
    
    highLow = pd.DataFrame({'high': df[computeOver[0]],'low': df[computeOver[1]]})
    
    lackOfData = df.loc[df['lost-data'] == 1].index
    
    for lDi,lDim in zip(lackOfData,lackOfData[1:]):
        highLow.loc[lDi:lDim-1,'high'] = highLow.loc[lDi:lDim-1,'high'].rolling(span,min_periods=1).max()
        highLow.loc[lDi:lDim-1,'low'] = highLow.loc[lDi:lDim-1,'low'].rolling(span,min_periods=1).min()
    
    highLow.loc[lDim:,'high'] = highLow['high'].rolling(span,min_periods=1).max()    
    highLow.loc[lDim:,'low'] = highLow['low'].rolling(span,min_periods=1).min()
    
    df[oKey] = (100*(df[computeOver[2]] - highLow['low']))/(highLow['high'] - highLow['low'] + 1e-10)
    
    if computeSignal:
        signalKey = oKey+'-signal'
        df[signalKey] = 0
        df.loc[df[oKey]>80,signalKey] = 1
        df.loc[df[oKey]<20,signalKey] = -1
    
        computeEWM(df,key=signalKey,span=3)
    

def computeSTC(df: pd.DataFrame, MACD_timeFrames: list = [12, 26], K_timeFrame: int = 10):
    """
    Computes the STC (Schaff Trend Cycle) indicator for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the necessary columns.
    - MACD_timeFrames (list): The time frames for the MACD calculation. Default values are [5, 9]. Other common values are [23, 50], [21, 28].
    - K_timeFrame (int): The time frame for the %K calculation. Default is 10.
    """
    requiresLostData(df)
    
    eprice = pd.DataFrame({'MACD_FRAME1': df['close'],'MACD_FRAME2': df['close']})
    eprice['lost-data'] = df['lost-data']
    computeEWM(eprice,key='MACD_FRAME1',span=MACD_timeFrames[0]) #creates ewm1
    computeEWM(eprice,key='MACD_FRAME2',span=MACD_timeFrames[1]) #creates ewm2
    
    eprice['MACD'] = eprice['eMACD_FRAME1'] - eprice['eMACD_FRAME2']
    computeSO(eprice,computeOver=['MACD','MACD','MACD'],span=K_timeFrame,oKey='KMACD',computeSignal=False)
    # computeEWM(eprice,key='KMACD',span=3) #creates eKMACD
    computeEWM(eprice,key='KMACD',span=23) #creates eKMACD
    
    # df = divideWithZero(df,
    #                     num=eprice['MACD']-eprice['KMACD'],
    #                     div=eprice['eKMACD']-eprice['KMACD'],
    #                     outKey='STC')
    # df['STC'] = 100.0 * (eprice['MACD']-eprice['KMACD']) / (eprice['eKMACD']-eprice['KMACD'] + 1e-10)
    computeSO(eprice,computeOver=['eKMACD','eKMACD','eKMACD'],span=K_timeFrame,oKey='STC',computeSignal=False)

    df['STC'] = eprice['STC']
    computeEWM(df,key='STC',span=23) #creates eSTC

    df['STC-signal'] = 0

    STC_prev = df['eSTC'].shift(1)
    df.loc[(df['eSTC'] > 25) & (STC_prev <= 25), 'STC-signal'] = 1
    df.loc[(df['eSTC'] < 75) & (STC_prev >= 75), 'STC-signal'] = -1

    # computeEWM(df,key='STC-signal')
    

def computeBB(df: pd.DataFrame, nSD: float = 2.0, span: int = 20):
        """
        Compute Bollinger Bands for a given DataFrame with additional columns 'BOLU', 'BOLD', and 'BB' representing the upper band,
            lower band, and Bollinger Band signal, respectively

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing 'high', 'low', 'close', and 'lost-data' columns.
        - nSD (float): Number of standard deviations for Bollinger Bands. Default is 2.0.
        - span (int): Number of periods to consider for rolling calculations. Default is 20.
        """
        requiresLostData(df)
        
        tprice = pd.DataFrame({'typicalPrice': (df['high']+df['low']+df['close'])/3.0})
        
        tprice['lost-data'] = df['lost-data']
        roll(tprice,key='typicalPrice',span=span) # creates AtypicalPrice
        tprice['std'] = 0.0
        
        lackOfData = df.loc[df['lost-data'] == 1].index
        for lDi,lDim in zip(lackOfData,lackOfData[1:]):
                tprice.loc[lDi:lDim-1,'std'] = tprice.loc[lDi:lDim-1,'typicalPrice'].rolling(span,min_periods=1).std()
        
        tprice.loc[lDim:,'std'] = tprice['typicalPrice'].rolling(span,min_periods=1).std() 
        
        df['BOLU'] = tprice['AtypicalPrice'] + nSD * tprice['std']
        df['BOLD'] = tprice['AtypicalPrice'] - nSD * tprice['std']

        df['BB'] = 0
        
        df.loc[(df['close'] > df['BOLU']) | (df['close'] < df['BOLD']),'BB'] = 1
        

# Indicator class including as methods the functions above
class Indicators:
    def __init__(self, data: Union[pd.DataFrame, str, int, None] = None ):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = loadData(key = data)
        elif isinstance(data, int):
            self.data = sampleData(dataIndex= data)
        else:
            self.data = sampleData()

        self.lostData()
        self.signals = {}

    def lostData(self):
        computeLostData(self.data)
    
    def OBV(self):
        computeOBV(self.data)
        self.signals['OBV'] = ['OBV','OBV-price-divergence','OBV-sign']

    def ATR(self, n=14):
        computeATR(self.data,n)
        self.signals['ATR'] = ['ATR']
    
    def AD(self):
        computeAD(self.data)
        self.signals['AD'] = ['AD','AD-price-divergence','AD-sign']
    
    # def ad_vs_price(self, M=14):
        # self.data = computeAD(self.data)
        # self.roll(self.data,'AD',span=M)
        # self.data['AD-price-divergence'] = (np.sign(self.data['dclose-sign']*self.data['AD-sign']) == -1)
        # self.data.loc[self.data['lost-data'] == 1,'AD-price-divergence'] = False
    
    def ADX(self):
        computeADX(self.data)
        self.signals['ADX'] = ['ADX','ADI','ADI2']

    def Aroon(self):
        computeAroon(self.data)
        self.signals['Aroon'] = ['aroon-up','aroon-down']
        # self.signals['Aroon'] = ['Aaroon-up','Aaroon-down']

    def MACD(self):
        computeMACD(self.data)
        self.signals['MACD'] = ['MACD','MACD-price-divergence']

    def RSI(self, n=14):
        computeRSI(self.data,n)
        self.signals['RSI'] = ['RSI','RSI-signal','RSI-price-divergence']

    def SO(self):
        computeSO(self.data)
        self.signals['SO'] = ['SO-signal', 'eSO-signal']


    def STC(self):
        computeSTC(self.data)
        self.signals['STC'] = ['STC-signal']
    
    def BB(self):
        computeBB(self.data)
        self.signals['BB'] = ['BOLU','BOLD','BB']

    def toMPLfinance(self):
        newDf = self.data.copy()
        newDf.rename(columns={'open': 'Open', 
                       'close': 'Close', 
                       'high': 'High', 
                       'low': 'Low', 
                       'volume': 'Volume',
                       'dtime': 'DateTime'},
                        inplace=True)
        newDf['DateTime'] = pd.to_datetime(newDf['DateTime'])
        newDf.set_index('DateTime', inplace=True)

        return newDf
    
    def runAll(self):
        self.OBV()
        self.ATR()
        self.AD()
        self.ADX()
        self.Aroon()
        self.MACD()
        self.RSI()
        self.SO()
        self.STC()
        self.BB()

    def plotIndicator(self, indicator: str, until: int = None):
        self.__getattribute__(indicator)()
        nSignals = len(self.signals[indicator])

        if until is not None:
            data = self.data.iloc[:until]

        # print(data.columns)

        # Create a new figure
        # fig = plt.figure(figsize=(10, 6))
        _, axes = plt.subplots(nSignals, 1, sharex=True, figsize=(10, 6))
        type(axes)
        if type(axes) is not np.ndarray:
            axes = np.array([axes])

        x_coords = data.time[data['lost-data'] == 1]
        for signal, ax in zip(self.signals[indicator],axes):
            ax.plot(data['time'], data['close'],':', color='b', zorder = 4)
            ylim = ax.get_ylim()

            # Create a twin axes
            axes_r = ax.twinx()

            # Plot the data on the second vertical axis
            axes_r.plot(data['time'], data[signal],'.', color='g',  zorder = 1, markersize=1)

            # Add vertical lines at the x-coordinates where 'lost-data' is True
            ax.vlines(x=x_coords, ymin=ylim[0], ymax=ylim[1], color='r', linestyle='--', zorder = 3)

            ax.set_ylabel(signal)
            
        xticks = ax.get_xticks()   
        closest_index = [abs(data['time']-x).idxmin() for x in xticks]
        ax.set_xticks(data['time'].loc[closest_index])
        ax.set_xticklabels(data['dtime'].loc[closest_index], rotation=45)

        # Show the figure
        plt.suptitle(indicator)
        plt.show()

    # def adx_trend(self):
        # self.data = computeADX(self.data)
        # self.data['adx_trend'] = np.where(self.data['ADX'] > 40, 'strong_trend',
                                        #   np.where(self.data['ADX'] < 20, 'weak_trend', 'range'))
    
