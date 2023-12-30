#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import io
get_ipython().run_line_magic('cd', "'/Users/rajeshprabhakarkaila/Desktop/Datasets'")


# In[3]:


SFair=pd.read_csv("Air_Traffic_Passenger_Statistics.csv")


# In[5]:


SFair.head()


# In[6]:


# Time Series Forecasting or IOT Data forecasting
# Time Series data has datetime or date variable which is the most 
# critical variable.
# In multivariate data, variables like weekday, month, quarter,year are
# extracted and date variable is deleted.

# Default Dateformat "YYYY-mm-dd" or datetimeformat "YYYY-mm-dd HH:MM:SS"
# 3 Types of Time Series data
# 1) Univariate Timeseries - Date & y(numeric)  - Only 2 variables
# 2) Multivariate Timeseries - Along with independent variables extracted
# from date other independent variables will also be there. Regression
# Algorithms are used
# 3) Panel Data - Along with Date and other independent variables an ID
# variable will also be there (Country, Company, Region)


# In[8]:


# Create Univariate Time Series from data
monthly_airtraffic=SFair[['Activity Period','Passenger Count']]


# In[12]:


monthly_airtraffic.info()
# Convert date into dateformat


# In[11]:


monthly_airtraffic['Activity Period']=pd.to_datetime(
    monthly_airtraffic['Activity Period'],format="%Y%m")
# %Y - YYYY & %y - YY & %m - mm & %M - month in text % %d -dd
# pd.to_datetime() - dateformat should be given according to existing form


# In[14]:


# Resample Timeseries into different Time frequencies
# pandas.reasmple()
# "60Min" - hour, "D"- Daily , "M"- Monthly, "Q"-Quarterly,"A"- Annual
# Statistical Function like sum(),mean(),median() or std() must be given
# After resampling Date will be automatically indexed into Rows
# In Time Series Date will be indexed into Row Numbers


# In[15]:


monthly_airtraffic=monthly_airtraffic.resample("M",
                                               on="Activity Period").sum()


# In[67]:


monthly_airtraffic.tail() # In Timeseries Row Indexing of Date is must


# In[17]:


# Time Series Plot - Line Plot
monthly_airtraffic.plot(kind="line")


# In[18]:


# Univariate Timeseries forcasting - Data must be stationary. Stationary
# means constant mean, constant variance and constant covariance.
# Typically assumption is Time should not effect data.
# Timeseries with trends and seasonality are not stationary data as time
# will effect changes at different time points.
# Trends - Up, Down, Neutral or Horizontal
# Seasonality - Based on 4 seasons - Summer, Rainy, Spring and Winter


# ![image.png](attachment:image.png)

# In[19]:


# Augmented Dickey Fuller Test of Stationarity - identifies whether data
# is stationary or not
# Null - Unit Root Present or Data is not Stationary
# Alt - No Unit Root or Data is Stationary

# Interpretation of test is based on p-value
# p-value < 0.05, Reject Null & p-value > 0.05, Fail to Reject Null


# In[20]:


from statsmodels.tsa.stattools import adfuller


# In[21]:


adfuller(monthly_airtraffic)
# Since p-value-0.04738274895948845 is less than 0.05, Reject Null


# In[22]:


# Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
# Null hypothesis that x is level or trend stationary.
# Alt hypothesis that x is not level or trend stationary.

# p-value must be greater than 0.05, Fail to Reject (Accept Null)


# In[23]:


from statsmodels.tsa.stattools import kpss


# In[24]:


kpss(monthly_airtraffic)
# Since p-value-0.1 is greater than 0.05, Fail to Reject Null


# In[29]:


# If Data is not stationary, differencing must be done using lags
# y          - 1,2,3,4,5,6,7,8,9
# lag 1 of y -   1,2,3,4,5,6,7,8,9
# First order diff = y - lag 1 of y 
# lag 2 of y -     1,2,3,4,5,6,7,8,9
# Second order diff = y - lag 2 of y

# .diff() - Differencing function - by deafult first order differencing

# After differencing repeat ADF test and KPSS test on differenced data


# In[26]:


monthly_airtraffic_diff=monthly_airtraffic.diff()


# In[27]:


monthly_airtraffic_diff=monthly_airtraffic_diff.dropna()


# In[28]:


monthly_airtraffic_diff.plot(kind="line")


# In[31]:


# Decomposition of Timeseries - Breaking Timeseries into components
# a) Trend - Up, Down, Neutral or Horizontal
# b) Seasonality - Based on 4 Seasons
# c) Cyclicality - Based on Business Cycles (Longterm Trend 8-12 years)
# d) Random or Residual or Error = y - lagged value of y

# Holt - Winters Method of Decomposition
# Additive Model (non Seasonal Data) = yt=Tt+St+Ct+Et
# Multiplicative Model (Seasonal Data) = yt= Tt*St*Ct*Et


# ![image.png](attachment:image.png)

# In[32]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[33]:


seasonal_decompose(monthly_airtraffic).plot()


# In[35]:


# Univariate Timeseries forecasting - Many methods like Simple Moving 
# Average, Exponential Moving Average, Holt-Winters method, etc. But
# most important is ARIMA forecasting

# Auto Regressive Integrated Moving Average (ARIMA) is a multiple linear
# regression model with 2 equations for forecasting future timeperiods
# based on historical timeperiods.

# Non Seasonal ARIMA - ARIMA(p,d,q) - p,d,q are lags between 0,0,0 till
# 5,2,5.
# Seasonal ARIMA - SARIMAX(p,d,q)[P,D,Q][time frequency]
# [P,D,Q][time frequency] are Seasonal parameters

# AR(p) - Autoregressive is linear relationship between y and lagged value
# of y. Yt=Bo+B1yt-1+B2yt-2+B3yt-3+......+Bnyt-n (positive terms)
 
# I(d) - Integrated parameter which is the differencing to be done for
# bringing data to stationary

# MA(q) - Moving Average is linear relationship between errors and 
# lagged value of errors or residual or y-lag of y
#  Yt = B0 - B1et-1-B2et-2-B3et-3 -.......-Bnet-n (Negative Terms)

#In ARIMA identifying p,d,q values is most critical
# d is the differencing to be done to bring data to stationary
# ACF plot or Autocorrelation plot identifies the lag value q in MA(q)
# PACF plot of Partial Autocorrelation plot identifies lag value p in AR(p)

# Blue region or dotted lines region depicts 95% confidence level and
# indicates significance. Anything with in dotted lines or blue area
# statistically close to zero and anything above is statistcally close
# to non zero

# From ACF plot, the first lag at which there is a big change - value of q
# From PACF plot the first lag at which it becomes negative - value of p


# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)

# In[36]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[38]:


plot_acf(monthly_airtraffic,lags=20)


# In[39]:


plot_pacf(monthly_airtraffic)


# In[40]:


# Auto ARIMA is a function in pmdarima library that tries all combinations
# of p,d,q values from 0,0,0 till 5,2,5
# Identifies the best fit lag order combination based on lowest Alkaike's
# Information Criterion (AIC)
# By default SARIMAX model is build and if seasonal parameters are not
# generated then it is a non seasonal ARIMA model


# In[42]:


from pmdarima.arima import auto_arima


# In[43]:


arima_model=auto_arima(monthly_airtraffic)


# In[44]:


arima_model.summary()
# SARIMAX(0, 1, 0) - Non Seasonal Model
# AIC - 6137.387
# P>|z| must be less than 0.05


# In[48]:


# Ljung Box Test of Residuals
# Null - Model does not show Lack of fit or Model is Fine
# Alt - Model does show Lack of Fit and Model is not Fine

# p-value must be greater than 0.05, Fail to Reject (Accept) Null

# Since Prob(Q):	0.15 is greater than 0.05, Fail to Reject Null


# In[46]:


arima_model.predict(n_periods=36).plot(kind="line")


# In[47]:


arima_model.plot_diagnostics()


# In[50]:


from prophet import Prophet


# In[51]:


monthly_airtraffic_df=monthly_airtraffic.reset_index()


# In[52]:


monthly_airtraffic_df.columns=['ds','y']


# In[53]:


m=Prophet()
m.fit(monthly_airtraffic_df)


# In[60]:


future=m.make_future_dataframe(periods=36,freq='M')


# In[61]:


forecast=m.predict(future)


# In[62]:


m.plot(forecast)


# In[66]:


np.round(forecast[['ds','yhat','yhat_lower','yhat_upper']],2).tail(36)


# In[68]:


np.round(forecast[['yhat','yhat_lower','yhat_upper']],
         2).tail(36).plot(kind="line")


# In[ ]:




