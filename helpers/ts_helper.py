import statsmodels.api as sm
from statsmodels.tsa.api import adfuller

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from typing import Any

plt.style.use('fivethirtyeight')


def test_stationary(timeseries):
    """Test if values are stationary using fueller test.
    
    Args:
        :timeseries (any): a variable to hold timeseries values.
        
    Returns:
        :timeseries results (Any): plots of rollingmean, rolling standard deviation, consiting of regular values, smoothed values,
        anything to test stantionary of the given data inputs.
    """
    
    # moving average and moving standard deviation
    moving_average = timeseries.rolling(window=12).mean()
    moving_standard_deviation = timeseries.rolling(window=12).std()
    
    # plots comparing the value being testing for stationary, rolling mean, standard deviation ect.
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(14,10))
    plt.plot(timeseries,color="green",label="original")
    plt.plot(moving_average,color="red",label="Moving Average")
    plt.plot(moving_standard_deviation,color="blue",label="Moving STD")
    plt.title("Timeseries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    # adfuller values
    
    print("\nResults from Fueller Tests\n")
    
    
    df_test = adfuller(timeseries["FEDFUNDS"])
    
    # output from adfueller
    
    df_output = pd.Series(df_test[0:4],index=['Test Statistic','p-value',"'#Lags Used","Number of Observations Used"])
    
    # print the values
    
    for key,value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print(df_output)
    print("P-Value")
    print(df_test[1])
    print("\nLags Used\n")
    print(df_test[2])
    
    return None
    









