#!/usr/bin/env python
# coding: utf-8

# # LSTM with Initial Data using 5 features

# In[1]:


import numpy as np
# fetching dataset
# making dataframe
import pandas as pd
# for plotting graphs
import matplotlib.pyplot as plt
# we have used this to convert date in string in .csv file to datetime format(YY-MM-DD) for visualization
import datetime as dt
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# (1) EarlyStopping - Stop training when a monitored metric(all features together) has stopped improving
# (2) ReduceLROnPlateau - Reduce learning rate when a metric(all features together) has stopped improving.
# (3) ModelCheckpoint - is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) 
#                   at some interval, so the model or weights can be loaded later to continue the training from the state saved.
# (4) TensorBoard - saves train and validation files generated during the model training
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# StandardScaler-it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1
from sklearn.preprocessing import StandardScaler

# will make your plot outputs appear and be stored within the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('stock_price_dada_inverse.csv')
data.drop('Total Trade Quantity', axis=1, inplace=True)
data.drop('Turnover (Lacs)', axis=1, inplace=True)
data


# In[3]:


columns = list(data)[1:6]
date = list(data['Date'])
date = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date]


print('Number of features selected: {}'.format(columns))


# In[4]:


plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (20, 5)
data['Close'].plot(label = 'Open Price', color = 'blue', linewidth = 2.0)
data['Open'].plot(label = 'Close Price', linestyle = '--', linewidth = 2.0, color = 'orange')
plt.legend(loc = 'upper left')
plt.title('TATA BEVERAGES SIMPLE MOVING AVERAGE')
plt.show()


# In[5]:


# PREPROCESSING -converting the float values in columns to string
data = data[columns].astype(str)
for i in columns:
    for j in range(0, len(data)):
        data[i][j] = data[i][j].replace(',', '')

# converting the data back to from string to float
data = data.astype(float)

# Using multiple features (predictors)
# train_values will be used for normalization
# train_values is a 2D-array having only each row in each sub-array
# Rows: 1235 
# Columns: 5(Open, High, Low, Last, Close)
train_values = data.values

print('Shape of training set == {}.'.format(train_values.shape))
train_values


# In[6]:


# The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
# StandardScaler performs the task of Standardization. 
# Usually a dataset contains variables that are different in scale. 
# For e.g. an Employee dataset will contain AGE column with values on scale 20-70 
# and SALARY column with values on scale 10000-80000.
# As these two columns are different in scale, they are Standardized to have common scale while building machine learning model.
r = StandardScaler()
# fit_transform() is used on the training data so that we can scale the training data
train_scaling = r.fit_transform(train_values)

scaling = StandardScaler()
scaling.fit_transform(train_values[:, 0:1])


# In[7]:


# Creating a data structure with 90 timestamps and 1 output
# Creating lists(X_trian, y_train) for storing the data for training.
# training INPUT
X_train = []
# storing data of Close columns
# values we want to predict
# training OUTPUT
y_train = []

# So the LSTM will be using 90 past days data to make a prediction.
# And similarly we will be do the predictions for next 60 days.
n_future = 60   # Number of days we want to make the prediction in the future
n_past = 90     # Number of past days we want to use to predict the future


# len(train_scaling) = 1235
for i in range(n_past, len(train_scaling) - n_future +1):
    X_train.append(train_scaling[i - n_past:i, 0:data.shape[1] - 1])
    y_train.append(train_scaling[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# (1067, 90, 5) (rows, columns, features(High, Low, Last, Close))
print('X_train shape == {}.'.format(X_train.shape))
# (1067, 1) (rows, feature(Open))
print('y_train shape == {}.'.format(y_train.shape))


# In[8]:


# LSTM MODEL CREATION
# initializing the neural network based on LSTM 
model = Sequential()
# layer 1 with 64 Nodes 
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, data.shape[1]-1)))
# layer 2 with 10 Nodes
model.add(LSTM(units=10, return_sequences=False))
# Layer 3 is a Dropout Layer to avoid Overfitting
model.add(Dropout(0.25))
# Layer 4 is always the Dense layer 
# units = 1 since we are prdicting only one value i.e Open price
model.add(Dense(units=1, activation='linear'))
# useing adam optimizer
model.compile(optimizer = "adam", loss='mean_squared_error')


# In[9]:


get_ipython().run_cell_magic('time', '', "# %%time :- prints the wall time(total time to execute a program in a computer) for the entire cell\n# MODEL TRAINING\n# EarlyStopping - Stop training when a monitored metric has stopped improving.\n# monitor - quantity to be monitored.\n# min_delta - minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.\n# patience - number of epochs with no improvement after which training will be stopped.\n# ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving.\n# factor - factor by which the learning rate will be reduced.\nes = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)\nrlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)\nmcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)\n\ntb = TensorBoard('logs')\n\nhistory = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)")


# In[10]:


# storing the next 60 days date for predixtion of future opening prices
future_list = pd.date_range(date[-1], periods=n_future, freq='1d').tolist()

future_list_ = []
for this_timestamp in future_list:
    future_list_.append(this_timestamp.date())


# In[11]:


# Perform Prediction
train_predict = model.predict(X_train[n_past:])
future_predict = model.predict(X_train[-n_future:])


# In[12]:


# x i.e date in string
# returns datetime format of x
def timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


# In[13]:


# inverse the predictions to original values
predicted_train = scaling.inverse_transform(train_predict)
predicted_future = scaling.inverse_transform(future_predict)


# In[14]:


predicted_train.shape,predicted_future.shape


# In[15]:


# making a dataframe for the future and predicted trained vales
FUTURE_PRED = pd.DataFrame(predicted_future, columns=['Open']).set_index(pd.Series(future_list))
TRAIN_PRED = pd.DataFrame(predicted_train, columns=['Open']).set_index(pd.Series(date[2 * n_past + n_future -1:]))


# In[16]:


# 
TRAIN_PRED.index = TRAIN_PRED.index.to_series().apply(timestamp)
TRAIN_PRED.tail()


# In[17]:


FUTURE_PRED.index = FUTURE_PRED.index.to_series().apply(timestamp)
FUTURE_PRED.tail()


# In[18]:


df = pd.read_csv ("stock_price_dada_inverse.csv")
df = df.set_index('Date')
df = df[df.index >= '2014-09-24']
df.index = pd.to_datetime(df.index)
df


# In[19]:


dk = pd.read_csv('Tata Global Beverages Ltd Next 60 Days__.csv')
dk = dk.set_index('Date')
dm = pd.DataFrame(dk, columns=['Open']).set_index(pd.Series(future_list))
dm.tail()


# In[20]:


from pylab import rcParams
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2014-09-24'
plt.plot(TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:].index,TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
plt.plot(df.loc[START_DATE_FOR_PLOTTING:].index, df.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')
plt.plot(FUTURE_PRED.index, FUTURE_PRED['Open'], color='r', label='Predicted Stock Price')
plt.axvline(x = min(FUTURE_PRED.index), color='green', linewidth=2, linestyle='--')
plt.legend()


# In[21]:


Performance =  TRAIN_PRED['Open']-df['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/df['Open'])
net_Percentage_Original = Percentage.mean()
net_Percentage_Original


# In[22]:


Performance =  FUTURE_PRED['Open']-dm['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/dm['Open'])
net_Percentage_Original_future = Percentage.mean()
net_Percentage_Original_future


# # LSTM with 6 Features which Includes Simple Moving Average

# In[23]:


data = pd.read_csv('stock_price_dada_inverse.csv')
data.drop('Total Trade Quantity', axis=1, inplace=True)
data.drop('Turnover (Lacs)', axis=1, inplace=True)
data


# In[24]:


def SimpleMovingAverage(datas, window):
    sma = datas.rolling(window = window).mean()
    return sma

data['sma_20'] = SimpleMovingAverage(data['Open'], 20)
data.fillna(0)


# In[25]:


columns = list(data)[1:7]
date = list(data['Date'])
date = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date]


print('Number of features selected: {}'.format(columns))


# In[26]:


plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (20, 10)
data['Open'].plot(label = 'Open PRICES', color = 'blue')
data['sma_20'].plot(label = 'SMA 20d', linestyle = '--', linewidth = 2, color = 'red')
plt.legend(loc = 'upper left')
plt.title('TATA BEVERAGES SIMPLE MOVING AVERAGE')
plt.show()


# In[27]:


data = data[columns].astype(str)
for i in columns:
    for j in range(0, len(data)):
        data[i][j] = data[i][j].replace(',', '')

data = data.astype(float)

# Using multiple features (predictors)
train_values = data.values

print('Shape of training set == {}.'.format(train_values.shape))
train_values


# In[28]:


r = StandardScaler()
train_scaling = r.fit_transform(train_values)

scaling = StandardScaler()
scaling.fit_transform(train_values[:, 0:1])


# In[29]:


# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future

for i in range(n_past, len(train_scaling) - n_future +1):
    X_train.append(train_scaling[i - n_past:i, 0:data.shape[1] - 1])
    y_train.append(train_scaling[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


# In[30]:


model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, data.shape[1]-1)))
model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer = "adam", loss='mean_squared_error')


# In[31]:


get_ipython().run_cell_magic('time', '', "es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)\nrlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)\nmcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)\n\ntb = TensorBoard('logs')\n\nhistory = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)")


# In[32]:


future_list = pd.date_range(date[-1], periods=n_future, freq='1d').tolist()

future_list_ = []
for this_timestamp in future_list:
    future_list_.append(this_timestamp.date())


# In[33]:


train_predict = model.predict(X_train[n_past:])
future_predict = model.predict(X_train[-n_future:])


# In[34]:


def timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


# In[35]:


predicted_train = scaling.inverse_transform(train_predict)
predicted_future = scaling.inverse_transform(future_predict)


# In[36]:


predicted_train.shape,predicted_future.shape


# In[37]:


FUTURE_PRED = pd.DataFrame(predicted_future, columns=['Open']).set_index(pd.Series(future_list))
TRAIN_PRED = pd.DataFrame(predicted_train, columns=['Open']).set_index(pd.Series(date[2 * n_past + n_future -1:]))


# In[38]:


TRAIN_PRED.index = TRAIN_PRED.index.to_series().apply(timestamp)
TRAIN_PRED.tail()


# In[39]:


FUTURE_PRED.index = FUTURE_PRED.index.to_series().apply(timestamp)
FUTURE_PRED.tail()


# In[40]:


df = pd.read_csv ("stock_price_dada_inverse.csv")
df = df.set_index('Date')
df = df[df.index >= '2014-09-24']
df.index = pd.to_datetime(df.index)
df


# In[41]:


dk = pd.read_csv('Tata Global Beverages Ltd Next 60 Days__.csv')
dk = dk.set_index('Date')
dm = pd.DataFrame(dk, columns=['Open']).set_index(pd.Series(future_list))
dm.tail()


# In[42]:


from pylab import rcParams
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2014-09-24'
plt.plot(TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:].index,TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
plt.plot(df.loc[START_DATE_FOR_PLOTTING:].index, df.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')
plt.plot(FUTURE_PRED.index, FUTURE_PRED['Open'], color='r', label='Predicted Stock Price')
plt.axvline(x = min(FUTURE_PRED.index), color='green', linewidth=2, linestyle='--')
plt.legend()


# In[43]:


Performance =  TRAIN_PRED['Open']-df['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/df['Open'])
net_Percentage_SMA = Percentage.mean()
net_Percentage_SMA


# In[44]:


Performance =  FUTURE_PRED['Open']-dm['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/dm['Open'])
net_Percentage_SMA_future = Percentage.mean()
net_Percentage_SMA_future


# # LSTM with 8 features which includes SMA and Bollinger Bands Technical Indicators

# In[45]:


data = pd.read_csv('stock_price_dada_inverse.csv')
data.drop('Total Trade Quantity', axis=1, inplace=True)
data.drop('Turnover (Lacs)', axis=1, inplace=True)
data


# In[46]:


def SimpleMovingAverage(datas, window):
    sma = datas.rolling(window = window).mean()
    return sma

data['sma_20'] = SimpleMovingAverage(data['Open'], 20)


# In[47]:


def bollinger_bands(datas, sma, window):
    std = datas.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

data['UpperBand'], data['LowerBand'] = bollinger_bands(data['Close'], data['sma_20'], 20)
data.fillna(0)


# In[48]:


#Plotting Bolling Bands with the help of Upper ,Lower and Simple Moving average line which together constitute a Bollinger Band as seen in the visualized figure

plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (20, 10)
data['Open'].plot(label = 'Open PRICES', color = 'green')
data['UpperBand'].plot(label = 'UPPER BB 20', linestyle = '--', linewidth = 2, color = 'blue')
data['sma_20'].plot(label = 'MIDDLE BB 20', linestyle = '--', linewidth = 2, color = 'grey')
data['LowerBand'].plot(label = 'LOWER BB 20', linestyle = '--', linewidth = 2, color = 'red')
plt.legend(loc = 'upper left')
plt.title('TATA BEVERAGES BOLLINGER BANDS')
plt.show()


# In[49]:


columns = list(data)[1:9]
date = list(data['Date'])
date = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date]


print('Number of features selected: {}'.format(columns))


# In[50]:


data = data[columns].astype(str)
for i in columns:
    for j in range(0, len(data)):
        data[i][j] = data[i][j].replace(',', '')

data = data.astype(float)

# Using multiple features (predictors)
train_values = data.values

print('Shape of training set == {}.'.format(train_values.shape))
train_values=pd.DataFrame(train_values)
train_values.fillna(0)
train_values=train_values[train_values.index >= 19]
train_values.head(20)
train_values=train_values.values


# In[51]:


r = StandardScaler()
train_scaling = r.fit_transform(train_values)

scaling = StandardScaler()
scaling.fit_transform(train_values[:, 0:1])


# In[52]:


# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future

for i in range(n_past, len(train_scaling) - n_future +1):
    X_train.append(train_scaling[i - n_past:i, 0:data.shape[1] - 1])
    y_train.append(train_scaling[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


# In[53]:


model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, data.shape[1]-1)))
model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer = "adam", loss='mean_squared_error')


# In[54]:


get_ipython().run_cell_magic('time', '', "es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)\nrlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)\nmcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)\n\ntb = TensorBoard('logs')\n\nhistory = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)")


# In[55]:


future_list = pd.date_range(date[-1], periods=n_future, freq='1d').tolist()

future_list_ = []
for this_timestamp in future_list:
    future_list_.append(this_timestamp.date())


# In[56]:


train_predict = model.predict(X_train[n_past:])
future_predict = model.predict(X_train[-n_future:])


# In[57]:


def timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


# In[58]:


predicted_train = scaling.inverse_transform(train_predict)
predicted_future = scaling.inverse_transform(future_predict)


# In[59]:


predicted_train.shape,predicted_future.shape


# In[60]:


FUTURE_PRED = pd.DataFrame(predicted_future, columns=['Open']).set_index(pd.Series(future_list))
TRAIN_PRED = pd.DataFrame(predicted_train, columns=['Open']).set_index(pd.Series(date[2 * n_past + n_future -1+19:]))


# In[61]:


TRAIN_PRED.index = TRAIN_PRED.index.to_series().apply(timestamp)
TRAIN_PRED.tail()


# In[62]:


FUTURE_PRED.index = FUTURE_PRED.index.to_series().apply(timestamp)
FUTURE_PRED.tail()


# In[63]:


df = pd.read_csv ("stock_price_dada_inverse.csv")
df = df.set_index('Date')
df = df[df.index >= '2014-10-28']
df.index = pd.to_datetime(df.index)
df


# In[64]:


dk = pd.read_csv('Tata Global Beverages Ltd Next 60 Days__.csv')
dk = dk.set_index('Date')
dm = pd.DataFrame(dk, columns=['Open']).set_index(pd.Series(future_list))
dm.tail()


# In[65]:


from pylab import rcParams
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2014-09-24'
plt.plot(TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:].index,TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
plt.plot(df.loc[START_DATE_FOR_PLOTTING:].index, df.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')
plt.plot(FUTURE_PRED.index, FUTURE_PRED['Open'], color='r', label='Predicted Stock Price')
plt.axvline(x = min(FUTURE_PRED.index), color='green', linewidth=2, linestyle='--')
plt.legend()


# In[66]:


Performance =  TRAIN_PRED['Open']-df['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/df['Open'])
net_Percentage_SMA_BB = Percentage.mean()
net_Percentage_SMA_BB


# In[67]:


Performance =  FUTURE_PRED['Open']-dm['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/dm['Open'])
net_Percentage_SMA_BB_future = Percentage.mean()
net_Percentage_SMA_BB_future


# # LSTM with 10 features which includes SMA, Bollinger Bands and Stochastics Technical Indicators

# In[68]:


data = pd.read_csv('stock_price_dada_inverse.csv')
data.drop('Total Trade Quantity', axis=1, inplace=True)
data.drop('Turnover (Lacs)', axis=1, inplace=True)
data


# In[69]:


def SimpleMovingAverage(datas, window):
    sma = datas.rolling(window = window).mean()
    return sma

data['sma_20'] = SimpleMovingAverage(data['Open'], 20)


# In[70]:


def bollinger_bands(datas, sma, window):
    std = datas.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

data['UpperBand'], data['LowerBand'] = bollinger_bands(data['Close'], data['sma_20'], 20)


# In[71]:


def stochastics( dataframe, low, high, close, k, d ):

    lowest_minimum  = dataframe[low].rolling( window = k ).min()
    highest_maximum = dataframe[high].rolling( window = k ).max()

    dataframe['%k'] = 100 * (dataframe[close] - lowest_minimum)/(highest_maximum - lowest_minimum)
    dataframe['%d'] = dataframe['%k'].rolling(window = d).mean()

    return dataframe

stochs = stochastics( data, 'Low', 'High', 'Open', 14, 3 )
data.fillna(0)


# In[72]:


#Plotting Stochastic Oscillator with the help of %K and %D values and plotting a lower and upper reference lines of 80 and 20 mark

plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (20, 10)
def Stochastics_Plotting(name, price, k, d):
    axis1 = plt.subplot2grid((10, 1), (0,0), rowspan = 5, colspan = 1)
    axis2 = plt.subplot2grid((10, 1), (6,0), rowspan = 4, colspan = 1)
    axis1.plot(data['Open'], color = 'blue', label = 'STOCK CHART')
    axis1.set_title(f'{name} STOCK PRICE')
    axis2.plot(k, color = 'green', linewidth = 1.5, label = '%K')
    axis2.plot(d, color = 'orange', linewidth = 1.5, label = '%D')
    axis2.axhline(80, color = 'black', linewidth = 1, linestyle = '--')
    axis2.axhline(20, color = 'black', linewidth = 1, linestyle = '--')
    axis2.set_title(f'{name} STOCHASTICS VALUES')
    axis2.legend()
    plt.show()
    
Stochastics_Plotting('TATA BEVERAGES', data['Open'], data['%k'], data['%d'])


# In[73]:


columns = list(data)[1:11]
date = list(data['Date'])
date = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date]


print('Number of features selected: {}'.format(columns))


# In[74]:


data = data[columns].astype(str)
for i in columns:
    for j in range(0, len(data)):
        data[i][j] = data[i][j].replace(',', '')

data = data.astype(float)

# Using multiple features (predictors)
train_values = data.values

print('Shape of training set == {}.'.format(train_values.shape))
train_values=pd.DataFrame(train_values)
train_values.fillna(0)
train_values=train_values[train_values.index >= 19]
train_values.head(20)
train_values=train_values.values


# In[75]:


r = StandardScaler()
train_scaling = r.fit_transform(train_values)

scaling = StandardScaler()
scaling.fit_transform(train_values[:, 0:1])


# In[76]:


# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future

for i in range(n_past, len(train_scaling) - n_future +1):
    X_train.append(train_scaling[i - n_past:i, 0:data.shape[1] - 1])
    y_train.append(train_scaling[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


# In[77]:


model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, data.shape[1]-1)))
model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer = "adam", loss='mean_squared_error')


# In[78]:


get_ipython().run_cell_magic('time', '', "es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)\nrlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)\nmcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)\n\ntb = TensorBoard('logs')\n\nhistory = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)")


# In[79]:


future_list = pd.date_range(date[-1], periods=n_future, freq='1d').tolist()

future_list_ = []
for this_timestamp in future_list:
    future_list_.append(this_timestamp.date())


# In[80]:


train_predict = model.predict(X_train[n_past:])
future_predict = model.predict(X_train[-n_future:])


# In[81]:


def timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


# In[82]:


predicted_train = scaling.inverse_transform(train_predict)
predicted_future = scaling.inverse_transform(future_predict)


# In[83]:


predicted_train.shape,predicted_future.shape


# In[84]:


FUTURE_PRED = pd.DataFrame(predicted_future, columns=['Open']).set_index(pd.Series(future_list))
TRAIN_PRED = pd.DataFrame(predicted_train, columns=['Open']).set_index(pd.Series(date[2 * n_past + n_future -1+19:]))


# In[85]:


TRAIN_PRED.index = TRAIN_PRED.index.to_series().apply(timestamp)
TRAIN_PRED.tail()


# In[86]:


FUTURE_PRED.index = FUTURE_PRED.index.to_series().apply(timestamp)
FUTURE_PRED.tail()


# In[87]:


df = pd.read_csv ("stock_price_dada_inverse.csv")
df = df.set_index('Date')
df = df[df.index >= '2014-10-28']
df.index = pd.to_datetime(df.index)
df


# In[88]:


dk = pd.read_csv('Tata Global Beverages Ltd Next 60 Days__.csv')
dk = dk.set_index('Date')
dm = pd.DataFrame(dk, columns=['Open']).set_index(pd.Series(future_list))
dm.tail()


# In[89]:


from pylab import rcParams
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2014-09-24'
plt.plot(TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:].index,TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
plt.plot(df.loc[START_DATE_FOR_PLOTTING:].index, df.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')
plt.plot(FUTURE_PRED.index, FUTURE_PRED['Open'], color='r', label='Predicted Stock Price')
plt.axvline(x = min(FUTURE_PRED.index), color='green', linewidth=2, linestyle='--')
plt.legend()


# In[90]:


Performance =  TRAIN_PRED['Open']-df['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/df['Open'])
net_Percentage_SMA_BB_SO = Percentage.mean()
net_Percentage_SMA_BB_SO


# In[91]:


Performance =  FUTURE_PRED['Open']-dm['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/dm['Open'])
net_Percentage_SMA_BB_SO_future = Percentage.mean()
net_Percentage_SMA_BB_SO_future


# # LSTM with 10 features which includes SMA, Bollinger Bands and Stochastics Technical Indicators

# In[92]:


data = pd.read_csv('stock_price_dada_inverse.csv')
data.drop('Total Trade Quantity', axis=1, inplace=True)
data.drop('Turnover (Lacs)', axis=1, inplace=True)
data


# In[93]:


def SimpleMovingAverage(datas, window):
    sma = datas.rolling(window = window).mean()
    return sma

data['sma_20'] = SimpleMovingAverage(data['Open'], 20)


# In[94]:


def bollinger_bands(datas, sma, window):
    std = datas.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

data['UpperBand'], data['LowerBand'] = bollinger_bands(data['Close'], data['sma_20'], 20)


# In[95]:


def stochastics( dataframe, low, high, close, k, d ):

    lowest_minimum  = dataframe[low].rolling( window = k ).min()
    highest_maximum = dataframe[high].rolling( window = k ).max()

    dataframe['%k'] = 100 * (dataframe[close] - lowest_minimum)/(highest_maximum - lowest_minimum)
    dataframe['%d'] = dataframe['%k'].rolling(window = d).mean()

    return dataframe

stochs = stochastics( data, 'Low', 'High', 'Open', 14, 3 )


# In[96]:


def get_macd(dataframe, price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    dataframe['macd'] = pd.DataFrame(exp1 - exp2)
    dataframe['Signal'] =(dataframe['macd'].ewm(span = smooth, adjust = False).mean())
    dataframe['Histogram'] = (dataframe['macd'] - dataframe['Signal'])    
    return dataframe

data = get_macd(data,data['Close'], 26, 12, 9)
data.fillna(0)


# In[97]:


columns = list(data)[1:14]
date = list(data['Date'])
date = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in date]


print('Number of features selected: {}'.format(columns))


# In[98]:


data = data[columns].astype(str)
for i in columns:
    for j in range(0, len(data)):
        data[i][j] = data[i][j].replace(',', '')

data = data.astype(float)

# Using multiple features (predictors)
train_values = data.values

print('Shape of training set == {}.'.format(train_values.shape))
train_values=pd.DataFrame(train_values)
train_values.fillna(0)
train_values=train_values[train_values.index >= 19]
train_values.head(20)
train_values=train_values.values


# In[99]:


r = StandardScaler()
train_scaling = r.fit_transform(train_values)

scaling = StandardScaler()
scaling.fit_transform(train_values[:, 0:1])


# In[100]:


# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future

for i in range(n_past, len(train_scaling) - n_future +1):
    X_train.append(train_scaling[i - n_past:i, 0:data.shape[1] - 1])
    y_train.append(train_scaling[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


# In[101]:


model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, data.shape[1]-1)))
model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer = "adam", loss='mean_squared_error')


# In[102]:


get_ipython().run_cell_magic('time', '', "es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)\nrlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)\nmcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)\n\ntb = TensorBoard('logs')\n\nhistory = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)")


# In[103]:


future_list = pd.date_range(date[-1], periods=n_future, freq='1d').tolist()

future_list_ = []
for this_timestamp in future_list:
    future_list_.append(this_timestamp.date())


# In[104]:


train_predict = model.predict(X_train[n_past:])
future_predict = model.predict(X_train[-n_future:])


# In[105]:


def timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


# In[106]:


predicted_train = scaling.inverse_transform(train_predict)
predicted_future = scaling.inverse_transform(future_predict)


# In[107]:


predicted_train.shape,predicted_future.shape


# In[108]:


FUTURE_PRED = pd.DataFrame(predicted_future, columns=['Open']).set_index(pd.Series(future_list))
TRAIN_PRED = pd.DataFrame(predicted_train, columns=['Open']).set_index(pd.Series(date[2 * n_past + n_future -1+19:]))


# In[109]:


TRAIN_PRED.index = TRAIN_PRED.index.to_series().apply(timestamp)
TRAIN_PRED.tail()


# In[110]:


FUTURE_PRED.index = FUTURE_PRED.index.to_series().apply(timestamp)
FUTURE_PRED.tail()


# In[111]:


df = pd.read_csv ("stock_price_dada_inverse.csv")
df = df.set_index('Date')
df = df[df.index >= '2014-10-28']
df.index = pd.to_datetime(df.index)
df


# In[112]:


dk = pd.read_csv('Tata Global Beverages Ltd Next 60 Days__.csv')
dk = dk.set_index('Date')
dm = pd.DataFrame(dk, columns=['Open']).set_index(pd.Series(future_list))
dm.tail()


# In[113]:


from pylab import rcParams
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2014-10-28'
plt.plot(TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:].index,TRAIN_PRED.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')
plt.plot(df.loc[START_DATE_FOR_PLOTTING:].index, df.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')
plt.plot(FUTURE_PRED.index, FUTURE_PRED['Open'], color='r', label='Predicted Stock Price')
plt.axvline(x = min(FUTURE_PRED.index), color='green', linewidth=2, linestyle='--')
plt.legend()


# In[114]:


Performance =  TRAIN_PRED['Open']-df['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/df['Open'])
net_Percentage_SMA_BB_SO_MACD = Percentage.mean()
net_Percentage_SMA_BB_SO_MACD


# In[115]:


Performance =  FUTURE_PRED['Open']-dm['Open']
Performance = pd.DataFrame(Performance)
Performance['Open'] = Performance['Open'].abs()
Percentage = 100 - (Performance['Open']*100/dm['Open'])
net_Percentage_SMA_BB_SO_MACD_future = Percentage.mean()
net_Percentage_SMA_BB_SO_MACD_future


# # PERFORMANCE

# In[116]:


data = {'Trend Technique':['LSTM with ORIGINAL DATASET', 'LSTM with ORIGINAL DATASET+SMA', 'LSTM with ORIGINAL DATASET+SMA+BB', 'LSTM with ORIGINAL DATASET+SMA+BB+SO', 'LSTM with ORIGINAL DATASET+SMA+BB+SO+MACD'],
        'NET Performance':[net_Percentage_Original, net_Percentage_SMA, net_Percentage_SMA_BB, net_Percentage_SMA_BB_SO, net_Percentage_SMA_BB_SO_MACD],
       'NET future Performance':[net_Percentage_Original_future, net_Percentage_SMA_future, net_Percentage_SMA_BB_future, net_Percentage_SMA_BB_SO_future, net_Percentage_SMA_BB_SO_MACD_future]}
data = pd.DataFrame(data)
data


# In[117]:


data.to_csv("Result.csv")


# In[118]:


lis_name = []
lis_val = []
for i in range(len(data)):
  lis_name.append(data["Trend Technique"].iloc[i])
  lis_val.append(data["NET Performance"].iloc[i])
fig, ax = plt.subplots(figsize=(25,8))
bars = ax.bar(lis_name, lis_val, width=0.5)
for bar in bars:
  height = bar.get_height()
  label_x_pos = bar.get_x() + bar.get_width() / 2
  ax.text(label_x_pos, height, s=f'{height}', ha='center',
  va='bottom')


# In[119]:


lis_name = []
lis_val = []
for i in range(len(data)):
  lis_name.append(data["Trend Technique"].iloc[i])
  lis_val.append(data["NET future Performance"].iloc[i])
fig, ax = plt.subplots(figsize=(25,8))
bars = ax.bar(lis_name, lis_val, width=0.5)
for bar in bars:
  height = bar.get_height()
  label_x_pos = bar.get_x() + bar.get_width() / 2
  ax.text(label_x_pos, height, s=f'{height}', ha='center',
  va='bottom')


# In[ ]:




