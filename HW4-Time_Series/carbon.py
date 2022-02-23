import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error 
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./data/CO2.csv', skiprows=53, header=[1,2], na_values=-99.99, skipinitialspace=True)
df = df.dropna() # Drop NaN values

# Other imputation  methods include forward filling–fill missing values with previous values, and interpolation.

### Cleaning
#co2 = df[['Yr', 'Mn', 'CO2']] 
#cols = ['Yr', 'Mn', 'CO2 (ppm)']
co2 = df[['Yr', 'Mn', 'Date_float', 'CO2']] 
cols = ['Yr', 'Mn', 'Date', 'CO2 (ppm)']
co2.columns = cols
co2['Time'] = (co2.Yr - 1958) + (co2.Mn-1)/12
# co2['Time'] = (co2.Date - 1958) 

# Train / Test Split
df_train, df_test = train_test_split(co2_drop, test_size=0.2, shuffle=False)
train, test = train_test_split(co2_drop, test_size=0.2, shuffle=False)
X = co2_drop["Time"].values.reshape(-1,1)
y = co2_drop["CO2 (ppm)"]

### Regression
# Linear
X_train = train["Time"].values.reshape(-1, 1)
y_train = train["CO2 (ppm)"]
reg = LinearRegression().fit(X_train, y_train)
# reg.score(X, y) = 0.9764628754826995
# reg.coef_ = array([1.40668098])
# reg.intercept_ = 306.1012358066282 or 308.9954694936227

# polynomial fit with degree = 2, 3
model2 = np.poly1d(np.polyfit(train["Time"], y_train, 2)) # 0.01212 x^2 + 0.8021 x + 314.1
model3 = np.poly1d(np.polyfit(train["Time"], y_train, 3)) # -0.0001184 x^3 + 0.02096 x^2 + 0.6249 x + 314.9

# Make predictions
X_test = test["Time"].values.reshape(-1,1)
y_test = test["CO2 (ppm)"]
co2_pre = reg.predict(X_train)
co2_pred = reg.predict(X_test)
co2_prediction = reg.predict(X)
quad_pre = model2(X_train)
quad_pred = model2(X_test)
quad_prediction = model2(X)
cube_pre = model3(X_train)
cube_pred = model3(X_test)
cube_prediction = model3(X)
final_pred = model2(X_test) + p_hat.iloc[test["Mn"]-1].values.reshape(-1,1)

# Plot Regression
# Test
plt.scatter(X, y, color='black') 
plt.plot(X, co2_prediction, color='blue', linewidth=2)
plt.plot(X, quad_prediction, color='red', linewidth=2)
plt.plot(X, cube_prediction, color='green', linewidth=2)
plt.title("CO2 levels over time at Mauna Loa")
plt.xlabel("Time since Jan 15, 1958 (years)")
plt.ylabel("Atmospheric CO2 concentration (ppm)")
plt.show()

# Train accuracy
resid = y_train - co2_pre
quad_resid = y_train - quad_pre.reshape(1,-1)[0]
cube_resid = y_train - cube_pre.reshape(1,-1)[0]

# Plot Residual Error
# plt.scatter(X_test, residuals)
# plt.scatter(X_train, resid)
# plt.title("Residuals (R_linear)")
# plt.scatter(X_train, quad_resid)
# plt.title("Residuals (R_quadratic)")
plt.scatter(X_train, cube_resid)
plt.title("Residuals (R_cubic)")
plt.ylabel("Residual")
plt.xlabel("Time since Jan 15, 1958 (years)")
plt.show()

# Test accuracy
rmse = np.sqrt(mean_squared_error(y_test, co2_pred))                # 10.642011413148976 
mape = mean_absolute_percentage_error(y_test, co2_pred) * 100       # 2.4505275995447513 
quad_rmse = np.sqrt(mean_squared_error(y_test, quad_pred))          # 2.5028073719221453 (*)
quad_mape = mean_absolute_percentage_error(y_test, quad_pred) * 100 # 0.5322789167375521 (*)
cube_rmse = np.sqrt(mean_squared_error(y_test, cube_pred))          # 4.15152534176945 
cube_mape = mean_absolute_percentage_error(y_test, cube_pred) * 100 # 0.8487342853146738
final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))          # 1.1493602690794222 
final_mape = mean_absolute_percentage_error(y_test, final_pred) * 100 # 0.20859165947990566

# Fitting Periodic Signal
# De-trend
residual = train.copy()
residual["CO2 (ppm)"] = y_train.values.reshape(-1,1) - quad_pre
p_hat = residual[["Mn", "CO2 (ppm)"]].groupby('Mn').mean().copy()
# De-season
residual["CO2 (ppm)"] = residual["CO2 (ppm)"].values.reshape(-1,1) - p_hat.iloc[residual["Mn"]-1].values.reshape(-1,1)

# Plotting detrended data
plt.scatter(X_train, residual["CO2 (ppm)"], color='black')
# plt.title("CO2 levels over time at Mauna Loa (detrended)")
plt.title("CO2 levels over time at Mauna Loa (detrended and deseasoned)")
plt.xlabel("Time since Jan 15, 1958 (years)")
plt.ylabel("Atmospheric CO2 concentration (ppm)")
plt.show()

# Plotting Periodic Signal 
xticks = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
p_hat.index = xticks
plt.scatter(p_hat.index, p_hat)
plt.title("Periodic Signal of Atmospheric CO2 concentration (1958-2007)")
plt.xlabel("Month")
plt.ylabel("Average monthly residual of [CO2 concentration - quadratic trend] (ppm)")
plt.show()

# Plot Final Model: Periodic + Trend
final_model = model2(X) + p_hat.iloc[co2_drop["Mn"]-1].values.reshape(-1,1)
plt.plot(X, final_model, color="red", label='Final Model')
plt.scatter(X_train, train["CO2 (ppm)"], color='black', label='Train')
plt.scatter(X_test, test["CO2 (ppm)"], color='blue', label='Test')
plt.legend()
plt.title("Final model for CO2 levels over time at Mauna Loa (Quadratic trend and periodic season)")
plt.xlabel("Time since Jan 15, 1958 (months)")
plt.ylabel("Atmospheric CO2 concentration (ppm)")
plt.show()
