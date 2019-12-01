import pandas as pd
import numpy as np
import Calc
import matplotlib.pyplot as plt

import statsmodels
from statsmodels.tsa.stattools import adfuller
import datetime


def get_data_temperature():
    return pd.read_table("Data/daily-min-temperatures.txt",
                         sep=',')


def get_data_demand():
    return pd.read_csv("Data/Daily_Demand_Forecasting_Orders.csv",
                       sep=';')


def get_data_occupancy():
    return pd.read_table("Data/occupancy_data/datatest.txt",
                       sep=',')


def get_data_imports():
    return pd.read_csv("Data/_imports.csv",
                       sep=',')


if __name__ == '__main__':
    data = pd.read_csv("Data/VaR.csv")
    data.Data = data.Data.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
    s1 = pd.Series(data.PZU.values, index=data.Data)
    print(adfuller(s1.values, autolag='AIC')[1])
    d = 0.65
    thres = 0.01
    a = Calc.frac_diff_FFD(s1, d)
    print(adfuller(a.values, autolag='AIC')[1])


