import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def get_weights(d, size):
    w = [1]
    for k in range(1, size):
        w_temp = -(w[-1] / k) * (d - k + 1)
        w.append(w_temp)
    return np.array(w)


def get_weights_FFD(d, thres):
    w = [1]
    k = 1
    while abs(w[-1]) > thres:
        w_temp = -w[-1] / k * (d - k + 1)
        w.append(w_temp)
        k = k + 1
    return np.array(w[:-1])


def plot_weights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = get_weights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    plt.show()
    return


def fracDiff(series, d, thres=.01):
    w = get_weights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ﬁllna(method='fﬁll').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isﬁnite(series.loc[loc, name]):
                continue
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def frac_diff(series, d, thres=.01):
    w = get_weights(d, series.shape[0])
    w_temp = np.cumsum(abs(w))
    w_temp /= w_temp[-1]
    skip = w_temp[w_temp > thres].shape[0]
    series_F = series.fillna(method='ffill').dropna()
    df = pd.Series()
    for iloc in range(skip, series_F.shape[0]):
        loc = series_F.index[iloc]
        if not np.isﬁnite(series.loc[loc]):
            continue
        df[loc] = np.dot(w[-(iloc + 1):, :].T, series_F.loc[:loc])[0]
    return df


def frac_diff_FFD(series, d, thres=1e-5):
    w = get_weights_FFD(d, thres)
    width = len(w) - 1
    seriesF, df = series.ﬁllna(method='ffill').dropna(), pd.Series()
    for iloc1 in range(width, seriesF.shape[0]):
        loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
        if not np.isﬁnite(series[loc1]):
            continue
        df[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])
    return df


def calculate_frac_diff_results(s, d, thres):
    a = frac_diff_FFD(s, d, thres)
    adf = adfuller(a, autolag='AIC')[1]
    temp = s[s.index >= a.index[0]]
    c = np.corrcoef(temp, a)[0][1]
    return [adf, c]


def plot_stats(stats, full: bool):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax2 = ax1.twinx()
    ax1.plot(list(stats.keys()), [k[1] for k in stats.values()], 'g-')
    ax2.plot(list(stats.keys()), [k[0] for k in stats.values()], 'b-')

    ax1.set_xlabel('Differentiation')
    ax1.set_ylabel('Correlation')
    ax2.set_ylabel('P-value')
    plt.title("Correlation and P-value")
    if full:
        x_max = 2
    else:
        x_max = 1
    plt.hlines(0.05, xmin=0, xmax=x_max)
    plt.show()
