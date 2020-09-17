#!/usr/bin/env python
# coding: utf-8

#                        #### Exponential smoothing method for univariate time series forecasting.

# Exponential Smoothing is a time series forecasting method for univariate data, that can be extended to support data which has trend & seasonal components.
# 
# Exponential smoothing forecasting methods are similar in that a prediction is a weighted sum of past observations, but the model explicitly uses an exponentially decreasing weight for past observations. Specifically, past observations are weighted with a geometrically decreasing ratio.

#                                          #### Types of Exponential Smoothing
#                                          
# There are three main types of exponential smoothing time series forecasting methods.
# 
# Simple Exponential Smoothing
# Simple Exponential Smoothing, is a time series forecasting method for univariate data which does not consider the trend and seasonality in the input data while forecasting. The prediction is just the weighted sum of past observations.
# It requires a single parameter, called alpha (α), also called the smoothing factor.
# 
# This parameter controls the rate at which the influence of the observations at prior time steps decay exponentially. Alpha is often set to a value between 0 and 1. Large values mean that the model pays attention mainly to the most recent past observations, whereas smaller values mean more of the history is taken into account when making a prediction.
# 
# Hyper Parameter:
# 
# α – Smoothing factor for the level.
# Let’s see Python example for Simple exponential Smoothing.
# 
# For this example I am using the Airline passengers dataset prvided by Kaggle. You can download the dataset from there.

# In[1]:


#import all libraries
from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[2]:


#read csv file
df = pd.read_csv(r"C:\Users\NQE00254\Desktop\Power BI Reports\Data Science Courses\Python\AirPassengers.csv")

#convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

#Grouping the columns use below syntax

#grouping = df.groupby('Date')["Passengers"].sum().reset_index(name ='Sales')
df.index.freq = 'MS'

#read the dataframe
df


# In[3]:


#assign the train value
train_data = df.iloc[:120]

#assign the test value
test_data = df.iloc[120:]
#To get the information about the data
train_data.info()


# In[4]:


#import chart libraries
import plotly.graph_objs as go
import chart_studio.plotly as py

#plot monthly sales
plot_data = [
    go.Scatter(
        x=train_data['Date'],
        y=train_data['Passengers'],
    )
]
plot_layout = go.Layout(
        title='Train Analysis Chart'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[5]:


import plotly.graph_objs as go
import chart_studio.plotly as py
#plot monthly sales
plot_data = [
    go.Scatter(
        x=test_data['Date'],
        y=test_data['Passengers'],
    )
]
plot_layout = go.Layout(
        title='Test Analysis Chart'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[6]:


train_data['Passengers'].plot(legend=True,label='TRAIN')
test_data['Passengers'].plot(legend=True,label='TEST',figsize=(12,8));


# In[7]:


from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(train_data['Passengers'], period=12).plot();


# In[8]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span = 12 # The model will consider the last 12 months weighted average for forecasting
alpha = 2/(span+1)
model = SimpleExpSmoothing(train_data['Passengers']).fit(smoothing_level=alpha)
test_predictions = model.forecast(36).rename('SES Forecast')


# In[10]:


#Now Lets Plot the Predictions
train_data['Passengers'].plot(legend=True,label='TRAIN')
test_data['Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# The SimpleExponentialModel does not consider the trend and seasonality. It will just take the weighted average of past data and forecast that average for all testing data. That’s why you can observe a straight line as the prediction. This model is not much useful for us.

#                                              #### Double Exponential Smoothing
#                                              
# Double Exponential Smoothing is an extension to Simple Exponential Smoothing that explicitly adds support for trends in the univariate time series.
# 
# In addition to the alpha parameter for controlling smoothing factor for the level, an additional smoothing factor is added to control the decay of the influence of the change in trend called beta (β).
# 
# The method supports trends that change in different ways: an additive and a multiplicative, depending on whether the trend is linear or exponential respectively.
# 
# Double Exponential Smoothing with an additive trend is classically referred to as Holt’s linear trend model, named for the developer of the method Charles Holt.
# 
# Additive Trend: Double Exponential Smoothing with a linear trend.
# 
# Multiplicative Trend: Double Exponential Smoothing with an exponential trend.
# For longer range (multi-step) forecasts, the trend may continue on unrealistically. As such, it can be useful to dampen the trend over time.
# 
# Hyperparameters:
# 
# Alpha: Smoothing factor for the level.
# Beta: Smoothing factor for the trend.

# In[11]:


#Double Exponential Smoothening
from statsmodels.tsa.holtwinters import ExponentialSmoothing
double_model = ExponentialSmoothing(train_data['Passengers'],trend='add').fit()
test_predictions = double_model.forecast(36).rename('DES Forecast')


# In[12]:


#Now Lets Plot the Predictions
train_data['Passengers'].plot(legend=True,label='TRAIN')
test_data['Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# ###### Double Exponential Smoothing will consider only the trend and ignore the seasonality. Since the data has upward trend, the prediction is a straight line with positive Slope.

#                                                    ### Triple Exponential Smoothing
# Triple Exponential Smoothing is an extension of Double Exponential Smoothing that explicitly adds support for seasonality to the univariate time series.
# 
# This method is sometimes called Holt-Winters Exponential Smoothing, named for two contributors to the method: Charles Holt and Peter Winters.
# 
# In addition to the alpha and beta smoothing factors, a new parameter is added called gamma (γ) that controls the influence on the seasonal component.
# 
# As with the trend, the seasonality may be modeled as either an additive or multiplicative process for a linear or exponential change in the seasonality.
# 
# Additive Seasonality: Triple Exponential Smoothing with a linear seasonality.
# 
# Multiplicative Seasonality: Triple Exponential Smoothing with an exponential seasonality
# 
# Triple exponential smoothing is the most advanced variation of exponential smoothing and through configuration, it can also develop double and single exponential smoothing models.
# 
# Additionally, to ensure that the seasonality is modeled correctly, the number of time steps in a seasonal period (Period) must be specified. For example, if the series was monthly data and the seasonal period repeated each year, then the Period=12.
# 
# Hyperparameters:
# 
# Alpha: Smoothing factor for the level.
# Beta: Smoothing factor for the trend.
# Gamma: Smoothing factor for the seasonality.
# Trend Type: Additive or multiplicative.
# Seasonality Type: Additive or multiplicative.
# Period: Time steps in seasonal period.

# In[14]:


#Triple Exponential Smoothening
from statsmodels.tsa.holtwinters import ExponentialSmoothing
triple_model = ExponentialSmoothing(train_data['Passengers'],trend='add',seasonal='add',seasonal_periods=12).fit()
test_predictions = triple_model.forecast(36).rename('TES Forecast')


# In[15]:


#Now Lets Plot the Predictions
train_data['Passengers'].plot(legend=True,label='TRAIN')
test_data['Passengers'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION');


# In[18]:


test_predictions


# In[17]:


import plotly.graph_objs as go
import chart_studio.plotly as py
#plot monthly sales
plot_data = [
    go.Scatter(
        y=test_predictions
    )
]
plot_layout = go.Layout(
        title='Test Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[19]:


import plotly.graph_objs as go
import chart_studio.plotly as py
#plot monthly sales
plot_data = [
    go.Scatter(
        y=test_predictions
    )
]
plot_layout = go.Layout(
        title='Test Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[ ]:




