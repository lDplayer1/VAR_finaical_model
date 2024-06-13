#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# 使用原始字符串方式导入数据
data = pd.read_csv(r'C:\Users\qq264\Desktop\POILBREUSDM (1).csv')

# 查看前几行数据
print(data.head())
# 检查缺失值
print(data.isnull().sum())

# 插值处理缺失值
data = data.interpolate()

# 再次检查缺失值是否处理完毕
print(data.isnull().sum())
# 将日期列设置为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 查看处理后的数据
print(data.head())


# In[3]:


from statsmodels.tsa.api import VAR

# 构建VAR模型
model = VAR(data)

# 选择最佳滞后期
lag_order = model.select_order(maxlags=12)
print(lag_order.summary())

# 拟合模型
var_model = model.fit(lag_order.aic)
print(var_model.summary())


# In[4]:


import matplotlib.pyplot as plt

# 获取冲击响应函数
irf = var_model.irf(12
                   )

# 绘制冲击响应函数图
irf.plot(orth=False)
plt.show()


# In[5]:


import matplotlib.pyplot as plt

# 获取冲击响应函数
irf = var_model.irf(120
                   )

# 绘制冲击响应函数图
irf.plot(orth=False)
plt.show()


# In[1]:


import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 使用原始字符串方式导入数据
data = pd.read_csv(r'C:\Users\qq264\Desktop\POILBREUSDM (1).csv')

# 查看前几行数据
print(data.head())

# 检查缺失值
print(data.isnull().sum())

# 插值处理缺失值
data = data.interpolate()

# 再次检查缺失值是否处理完毕
print(data.isnull().sum())

# 将日期列设置为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 查看处理后的数据
print(data.head())

# 构建VAR模型
model = VAR(data)

# 选择最佳滞后期
lag_order = model.select_order(maxlags=12)
print(lag_order.summary())

# 拟合模型
var_model = model.fit(lag_order.aic)
print(var_model.summary())

# 获取冲击响应函数
irf = var_model.irf(12)

variables = data.columns
for i in range(len(variables)):
    fig = irf.plot(response=variables[i], orth=False)
    fig.suptitle(f'Response of {variables[i]} to Shocks')
    plt.show()


# In[2]:


import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 使用原始字符串方式导入数据
data = pd.read_csv(r'C:\Users\qq264\Desktop\POILBREUSDM (1).csv')

# 查看前几行数据
print(data.head())

# 检查缺失值
print(data.isnull().sum())

# 插值处理缺失值
data = data.interpolate()

# 再次检查缺失值是否处理完毕
print(data.isnull().sum())

# 将日期列设置为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 查看处理后的数据
print(data.head())

# 构建VAR模型
model = VAR(data)

# 选择最佳滞后期
lag_order = model.select_order(maxlags=12)
print(lag_order.summary())

# 拟合模型
var_model = model.fit(lag_order.aic)
print(var_model.summary())

# 进行未来预测
forecast = var_model.forecast(data.values, steps=12)
forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=data.index[-1], periods=12, freq='M'), columns=data.columns)

# 输出预测结果
print("forcast")
print(forecast_df)


# In[3]:


# 绘制每个变量的过去值和预测值的折线图
plt.figure(figsize=(12, 8))
for col in data.columns:
    plt.plot(data.index, data[col], label=f'Historical {col}')
    plt.plot(forecast_df.index, forecast_df[col], linestyle='--', label=f'Forecast {col}')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.title(f'Historical and Forecasted Values of {col}')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[1]:


import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import numpy as np

# 使用原始字符串方式导入数据
data = pd.read_csv(r'C:\Users\qq264\Desktop\POILBREUSDM (1).csv')

# 插值处理缺失值
data = data.interpolate()

# 将日期列设置为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 将数据划分为训练集和测试集
train = data.iloc[:-12]
test = data.iloc[-12:]

# 构建VAR模型
model = VAR(train)

# 选择最佳滞后期
lag_order = model.select_order(maxlags=12)

# 拟合模型
var_model = model.fit(lag_order.aic)

# 进行预测
forecast = var_model.forecast(train.values, steps=12)
forecast_df = pd.DataFrame(forecast, index=test.index, columns=data.columns)

# 计算RMSE
rmse = np.sqrt(mean_squared_error(test.values, forecast_df.values))
print("Root Mean Squared Error (RMSE):", rmse)


# In[7]:


#回归尝试
df=data
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# 定义逐步回归函数
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# 逐步回归选择显著变量
X = df.drop(columns=['oil_price'])  # 以oil_price为例
y = df['oil_price']
resulting_vars = stepwise_selection(X, y)
print('Resulting variables:', resulting_vars)


# In[4]:



# 筛选出2023年之后的数据
filtered_data = data[data.index >= '2023-01-01']

# 绘制每个变量的过去值和预测值的折线图
plt.figure(figsize=(12, 8))
for col in data.columns:
    plt.plot(filtered_data.index, filtered_data[col], label=f'Historical {col}')
    plt.plot(forecast_df.index, forecast_df[col], linestyle='--', label=f'Forecast {col}')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.title(f'Historical and Forecasted Values of {col}')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[9]:


# 计算数据集的平均值
rmse=14.556817179134292

# 计算相对误差
relative_error = (rmse / data_mean) * 100
print("Relative Error (%):", relative_error)


# In[ ]:




