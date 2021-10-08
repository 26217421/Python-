import pandas as pd
import numpy as np
import datetime

from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    stock_df = pd.read_csv('D600000.csv')
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.set_index('date', inplace=True)
    #
    stock_df = stock_df['close']
    stock_df = stock_df['2017':'2021']

    stock_diff = stock_df - stock_df.shift(1)
    stock_diff.dropna(inplace=True)

    lag_pacf = pacf(stock_diff, nlags=20)
    lag_acf = acf(stock_diff, nlags=20)

    # Plot PACF:
    plt.subplot(121)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(stock_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(stock_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    # Plot ACF:
    plt.subplot(122)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(stock_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(stock_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    plt.show()

    stock_df = stock_df.dropna()
    model = ARIMA(stock_df, order=(6, 1, 6))
    result = model.fit()
    print(result.summary())

    print('之后20天预测')
    pred = result.forecast(15)
    pred = pd.Series(pred[0], index=pd.period_range('20210928', '20211017', freq='B'))
    print(pred)
    plt.figure(figsize=(20, 8))
    plt.plot(stock_df)
    plt.plot(pred)
    plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.show()

    ts = result.predict()
    df_fill = stock_df[ts.index]
    print('RMSE:{}'.format(np.sqrt(sum((ts - df_fill) ** 2) / ts.size)))

    #残差校验
    resid = result.resid  # 赋值
    #Durbin-Watson）检验
    '''
    当DW值显著的接近于O或４时，则存在自相关性，而接近于２时，则不存在（一阶）自相关性。这样只要知道ＤＷ统计量的概率分布，在给定的显著水平下，根据临界值的位置就可以对原假设Ｈ０进行检验
    '''
    print(sm.stats.durbin_watson(result.resid.values))
    # 白噪声检验
    '''
    时间序列中一个最基本的模型就是高斯白噪声序列。而对于ARIMA模型，
    其残差被假定为高斯白噪声序列，所以当我们用ARIMA模型去拟合数据时，
    拟合后我们要对残差的估计序列进行LB检验，判断其是否是高斯白噪声，
    如果不是，那么就说明ARIMA模型也许并不是一个适合样本的模型。
    '''
    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


