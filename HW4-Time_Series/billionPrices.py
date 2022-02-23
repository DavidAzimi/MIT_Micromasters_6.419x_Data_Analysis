import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error 
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg

### Import Data
# consumer price index, the price of a "market basket of consumer goods and services" - a proxy for inflation
cpi = pd.read_csv('./data/PriceStats_CPI.csv', parse_dates=["date"])
# break-even rate, the difference in yield between a fixed rate and inflation adjusted 10 year treasury note
ber = pd.read_csv('./data/T10YIE.csv', parse_dates=["DATE"])

### Duplicate and/or Missing
# cpi.index.duplicated().sum()  0
# There are no duplocate indeces
# cpi.duplicated().any()  True
# cpi.duplicated().sum()  22
# cpi[cpi.duplicated(keep=False)]
# Looking at duplicates indicates equal values a few days apart. For the purposes of this analysis, these can be safely removed (we are looking at the first day of each month of each year) 
cpi = cpi.drop_duplicates()
ber = ber.drop_duplicates()
# We can drop NAs but may want to consider imputation
cpi = cpi.dropna()
ber = ber.dropna()

cols = ['date', 'T10YIE']
ber.columns = cols

# check values and distribution
dfs = [ber, cpi]
for df in dfs:
    df.head()
    df.shape
    df.date.min()
    df.date.max()
    df.isna().sum()
"""
ber     date  T10YIE              #   cpi     date  PriceStats    CPI
0 2003-01-02    1.64              #   0 2008-07-24   100.00000  100.0
1 2003-01-03    1.62              #   1 2008-07-25    99.99767  100.0
2 2003-01-06    1.63              #   2 2008-07-26    99.92376  100.0
3 2003-01-07    1.62              #   3 2008-07-27    99.91537  100.0
4 2003-01-08    1.71              #   4 2008-07-28    99.89491  100.0
(4215, 2)                         #   (4087, 3) 
Timestamp('2003-01-02 00:00:00')  #   Timestamp('2008-07-24 00:00:00')
Timestamp('2019-11-04 00:00:00')  #   Timestamp('2019-10-01 00:00:00')
date        0                         date           0
T10YIE      0                         PriceStats     0
                                      CPI            0 
"""
### Features

# Seasonal averages
monthly_CPI = cpi.groupby(cpi['date'].dt.month)['CPI'].mean()
monthly_PS = cpi.groupby(cpi['date'].dt.month)['PriceStats'].mean()
monthly_T10 = ber.groupby(ber['date'].dt.month)['T10YIE'].mean()

# Monthly plots
cpi = cpi.set_index('date')
cpi['PSshift']=cpi['PriceStats'].shift(1)
cpi['dir'] = (cpi['PriceStats'] - cpi['PSshift']) / cpi['PSshift']
g=cpi.groupby(pd.Grouper(freq="M")).nth(0) # First of the month
g2=cpi.groupby(pd.Grouper(freq="M")).mean() # Average
g3=cpi.groupby(pd.Grouper(freq="M")).nth(-1) # Last of month 
g2['mir'] = g2['dir'] * 30
cpi_t = g['CPI']
plt.plot(cpi_t)

ber['T10YIE'] = ber['T10YIE'] / 100
ber = ber.set_index('date')
b = ber.groupby(pd.Grouper(freq="M")).mean()
b['monthly'] = (b['T10YIE']+1)**(1/12)-1

'''
In the autograded section of this assignment we trained our models to predict CPI
from data up until Sep 2013. 
For the report we trained models to predict Monthly Inflation Rate.
'''

# Train/Test Split
idx =[i for i in range(len(g))]
g['date'] = g.index
g.index = idx
X = g.index.values.reshape(-1,1)
y = g['CPI'].values
cpi_train = g[g.date < "2013-09-01"]
X_train = cpi_train.index.values.reshape(-1,1)
y_train = cpi_train['CPI'].values
cpi_test = g[g.date >= "2013-09-01"]
X_test = cpi_test.index.values.reshape(-1,1)
y_test = cpi_test['CPI'].values

# Report split
g = g.dropna()
idx =[i for i in range(len(g))]
g['date'] = g.index
g.index = idx
X = g.index.values.reshape(-1,1)
y = g['MIR'].values
mri_train = g[g.date < "2013-09-01"]
mri_test = g[g.date >= "2013-09-01"]
X_train = mri_train.index.values.reshape(-1,1)
y_train = mri_train['MIR'].values
X_test = mri_test.index.values.reshape(-1,1)
y_test = mri_test['MIR'].values


# Detrend CPI_t = T_t + R_t
reg = LinearRegression().fit(X_train, y_train)
# reg.coef_  array([0.16104348])
# reg.intercept_ 96.72932632872502
plt.scatter(X_train, y_train, color='black')
plt.scatter(X_test, y_test, color='blue')
plt.plot(X, reg.predict(X), color='red', lw=2)
plt.show()

# Residuals
R_t = y_train - reg.predict(X_train)
full_res = y - reg.predict(X)
plt.title('Detrended training data')
plt.scatter(X_train, R_t)
plt.show()

# AR Model
plot_pacf(R_t) # suggests an order p = 2
plot_acf(R_t)
R = cpi_train['CPI']
R = R - reg.predict(X_train)
R = R.to_frame()
R['shift'] = R['CPI'].shift()
R['shift2'] = R['shift'].shift()
R.dropna(inplace=True)
X_t = R['CPI'].values
X_1 = R['shift'].values
X_2 = R['shift2'].values

res = AutoReg(R_t, lags = 2, trend = 'n').fit()
res_gen = AutoReg(full_res, lags = 2, trend = 'n')
out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
print(out.format(res.aic, res.hqic, res.bic))
plt.scatter(X_train[2:], R['CPI'], color='black')
plt.plot(X_train[2:], res.params[0]*X_1 + res.params[1]*X_2, color='red', lw=2)
plt.plot(X_train[3:], res.fittedvalues, color='blue', lw=2)
plt.show()

# Predictions
trend = reg.predict(X_test)
residuals = res_gen.predict(res.params, start=len(y_train), end=len(y_train)+len(y_test)-1, dynamic=False)
y_pred = trend + residuals
y_prediction = reg.predict(X[2:])+res_gen.predict(res.params, start=0, end=len(y_train)+len(y_test)-1, dynamic=False)

plt.scatter(X_train, y_train, color='black')
plt.scatter(X_test, y_test, color='blue')
plt.plot(X[2:], y_prediction, color='red', lw=2)
plt.show()

#Accuracy
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Monthly inflation rate
g['shift']=g['CPI'].shift(1)
g['PSshift']=g['PriceStats'].shift(1)
g['MIR'] = (g['CPI'] - g['shift']) / g['shift']
g['logIR'] = np.log(g['CPI']) - np.log(g['shift'])
g['psIR1'] = (g['PriceStats'] - g['PSshift']) / g['PSshift']
g = g.dropna() # shift removes a row

g2['shift']=g2['CPI'].shift(1)
g2['MIR'] = (g2['CPI'] - g2['shift']) / g2['shift']
g2['PSshift']=g2['PriceStats'].shift(1)
g2['psIR1'] = (g2['PriceStats'] - g2['PSshift']) / g2['PSshift']

g3['shift']=g3['CPI'].shift(1)
g3['MIR'] = (g3['CPI'] - g3['shift']) / g3['shift']
g3['PSshift']=g3['PriceStats'].shift(1)
g3['psIR1'] = (g3['PriceStats'] - g3['PSshift']) / g3['PSshift']

plt.plot(g['date'], g['MIR'], color='red', lw=2)

# Inflation Rate Plot
fig, ax = plt.subplots()
ax.plot(g.index, g['MIR']*100, color='black', lw=2)
ax.axhline(y=g['MIR'].median()*100, color='blue', ls='dashed', label='median = 0.1245%')
# Major ticks every 6 months.
fmt_half_year = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(fmt_half_year)
# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)
# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Format the coords message box, i.e. the numbers displayed as the cursor moves
# across the axes within the interactive GUI.
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.
ax.grid(True)
# Rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them.
fig.autofmt_xdate()
plt.title('Monthly U.S. Inflation Rate as a Percentage (July 2008- October 2019)')
plt.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

# Inflation Rate Linear Trend Plot
fig, ax = plt.subplots()
ax.plot(X, reg.predict(X), color='red', lw=2)
ax.scatter(X_train, y_train, color='black', label='train')
ax.scatter(X_test, y_test, color='blue', label='test')
plt.title('Linear Trend of Monthly U.S. Inflation Rate as a Percentage (Training data is black)')
plt.xlabel('Months since Augst 2008')
plt.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
