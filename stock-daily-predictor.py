import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import yfinance as yf
from pandas_datareader import data as pdr
from stock_indicators import indicators, Quote
import mpl_finance as mpf
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import sys
yf.pdr_override()

from datetime import datetime

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

def score_model_history(model):
    model = model.history
    history_df = pd.DataFrame(model)

    history_df_filtered = history_df[history_df.index >= 150]
    max_val_loss = history_df_filtered['val_loss'].max()
    # Start the plot at epoch 5. You can change this to get a different view.
    # history_df.loc[:, ['loss', 'val_loss']].plot()
    print("Minimum Validation Loss: {}".format(history_df['val_loss'].min()))
    print("Max Validation Loss: {}".format(max_val_loss))
    print("Mean Validation Loss: {}".format(history_df_filtered['val_loss'].mean()))

    # plot = plt.show()
    # return plot

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

dtype = {
    'Close': 'float32',
}

btc_data_live = pdr.get_data_yahoo('AAL', start='2014-09-21', end=end)
# btc_data = pd.read_csv('BTC-USD-MAX.csv', dtype=dtype, parse_dates=['Date'])
btc_data = btc_data_live.copy()
btc_data.reset_index(inplace=True)
btc_data.set_index('Date', inplace=True)

future_pred = 3 # Number of days to forcast
# lag featuers
lags = 1
input_shape = [17]
random_state = 0

def create_feature_columns(dataset, y_value):
    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        dataset["Lag%s" % str(i+1)] = dataset[y_value].shift(i+1).pct_change()
        
    dataset['Prediction'] = dataset[y_value].shift(-future_pred) # creating new column called 'prediction' that contains the close price of the next 3 days
    dataset['Daily Return'] = dataset['Close'].pct_change() # creating new column called 'Daily Return' that contains the daily return of the stock
    dataset['volume_gap'] = dataset.Volume.pct_change()
    dataset['ho'] = dataset['High'] - dataset['Open'] 
    dataset['lo'] = dataset['Low'] - dataset['Open']
    dataset['oc'] = dataset.Open - dataset.Close
    dataset['hl'] = dataset.High - dataset.Low
    dataset['Dates'] = dataset.index
    dataset['day_of_week'] = dataset['Dates'].dt.dayofweek
    dataset['day_of_month'] = dataset['Dates'].dt.day
    quotes_list = [
        Quote(d,o,h,l,c,v)
        for d,o,h,l,c,v
        in zip(dataset['Dates'], dataset['Open'], dataset['High'], dataset['Low'], dataset['Close'], dataset['Volume'])
    ]

    results = indicators.get_macd(quotes_list)

    for r in results:
        dataset.loc[dataset['Dates'] == r.date, 'MACD'] = r.macd
        
    vol_ma = indicators.get_ema(quotes_list, 5)

    for r in vol_ma:
        dataset.loc[dataset['Dates'] == r.date, 'EMA'] = r.ema

    dataset['MACD'].fillna(0, inplace=True)

    dataset['EMA'].fillna(0, inplace=True)
    dataset.drop('Dates', axis=1, inplace=True)
    dataset.dropna(inplace=True) # dropping NaN values
    
    return dataset

def prepare_dataset(dataset, y_value='Prediction'):
    
    # create X and y
    X = dataset.drop('Prediction', axis=1)
    X_fcast = X[-future_pred:] # creating the set that we will use to predict the future values
    X = X[:-future_pred] # removing the last 3 rows
    y = dataset[y_value] # <==== UPDATE THIS FOR PREDICTED OUTCOME
    y = y[:-future_pred] # removing the last 3 rows
    
    # remove missing values and outliers
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]
    X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
    y = y[X.index]

    # splitting train/test
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=random_state)

    # create a scaler object
    scaler = StandardScaler()

    # fit the scaler on the training data
    scaler.fit(X_train)

    # transform the training and forecast data
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_fcast_scaled = scaler.transform(X_fcast)
    
    return X_train_scaled, X_valid_scaled, y_train, y_valid, X_fcast_scaled

# HIGH PREDICTIONS
dataset_high = btc_data.copy()
dataset_high.index = btc_data.index

create_feature_columns(dataset_high, 'High')

high_result = prepare_dataset(dataset_high)

# transform the training and forecast data
X_train_high_scaled = high_result[0]
X_valid_high_scaled = high_result[1]
X_fcast_high_scaled = high_result[4]
y_train_high = high_result[2]
y_valid_high = high_result[3]

# LOW PREDICTIONS
dataset_low = btc_data.copy()
dataset_low.index = btc_data.index

create_feature_columns(dataset_low, 'Low')

low_result = prepare_dataset(dataset_low)

# transform the training and forecast data
X_train_low_scaled = low_result[0]
X_valid_low_scaled = low_result[1]
X_fcast_low_scaled = low_result[4]
y_train_low = low_result[2]
y_valid_low = low_result[3]

# Close PREDICTIONS
dataset_close = btc_data.copy()
dataset_close.index = btc_data.index

create_feature_columns(dataset_close, 'Close')

close_result = prepare_dataset(dataset_close)

# transform the training and forecast data
X_train_close_scaled = close_result[0]
X_valid_close_scaled = close_result[1]
X_fcast_close_scaled = close_result[4]
y_train_close = close_result[2]
y_valid_close = close_result[3]

# X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
# X_valid_scaled = np.reshape(X_valid_scaled, (X_valid_scaled.shape[0], X_valid_scaled.shape[1], 1))
# X_fcast_scaled = np.reshape(X_fcast_scaled, (X_fcast_scaled.shape[0], X_fcast_scaled.shape[1], 1))

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=200, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# model = keras.Sequential([
#     layers.LSTM(256, activation='relu', return_sequences=True, input_shape=(7, 1)),
#     layers.LSTM(256, activation='relu', return_sequences=True),
#     layers.LSTM(256, activation='relu', return_sequences=True),
#     layers.Dense(1),
# ])

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

optimizer = model.compile(
    optimizer='adam',
    loss='mae',
)


def run_model(model, X_train_scaled, X_valid_scaled, y_train, y_valid, X_fcast_scaled, y_value, dataset):
    y_model = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_valid_scaled, y_valid),
        batch_size=256,
        epochs=500,
        callbacks=[early_stopping],
        verbose=0, # hide the output because we have so many epochs 
    )

    forcast = model.predict(X_fcast_scaled)
    print(score_model_history(y_model))
    forcast = forcast[-future_pred:] # select only the last 5 predictions
    forcast = np.reshape(forcast, (forcast.shape[0], forcast.shape[1])) # reshape into a 5x2 array

    # create dataframe for Low predictions
    d = dataset[[y_value]].tail(len(forcast))
    d = d.tail(future_pred)
    d.index = pd.date_range(start=btc_data.index[-1], periods=(len(d)), freq='D')
    prediction = pd.DataFrame(forcast)
    prediction.index = d.index
    prediction['Actual Price'] = btc_data_live[y_value]
    prediction.rename(columns = {0: 'Forecasted Price'}, inplace=True)

    return prediction


prediction_low = run_model(model, X_train_low_scaled, X_valid_low_scaled, y_train_low, y_valid_low, X_fcast_low_scaled, 'Low', dataset_low)
prediction_high = run_model(model, X_train_high_scaled, X_valid_high_scaled, y_train_high, y_valid_high, X_fcast_high_scaled, 'High', dataset_high)
prediction_close = run_model(model, X_train_close_scaled, X_valid_close_scaled, y_train_close, y_valid_close, X_fcast_close_scaled, 'Close', dataset_close)

# concatenate the two dataframes into a single dataframe
predictions = pd.concat([prediction_low, prediction_high, prediction_close], axis=1)
print(predictions)


import json

# your code for prediction

# convert prediction results to dictionary
prediction_dict = {'closing prediction': prediction_close.to_dict(),
                   'high prediction': prediction_high.to_dict(),
                   'low prediction': prediction_low.to_dict()}

# convert Timestamp objects to strings
for key in prediction_dict:
    for subkey in prediction_dict[key]:
        if isinstance(subkey, pd.Timestamp):
            prediction_dict[key][subkey.strftime('%Y-%m-%d')] = prediction_dict[key].pop(subkey)

# convert dictionary to JSON string
prediction_json = json.dumps(prediction_dict)

# print JSON string
print(prediction_json)


# history_df = pd.DataFrame(btc_data)
history_df = btc_data.copy()
history_df.index = pd.DatetimeIndex(history_df.index).astype(np.int64) // 10**9
history_df_filtered = history_df[history_df.index >= pd.Timestamp(150, unit='D').timestamp()]
# max_val_loss = history_df_filtered['val_loss'].max()
# # Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[:, ['Open', 'High', 'Low', 'Close']].plot()
# print("Minimum Validation Loss: {}".format(history_df['val_loss'].min()))
# print("Max Validation Loss: {}".format(max_val_loss))
# print("Mean Validation Loss: {}".format(history_df_filtered['val_loss'].mean()))

plt.show()




# prices = pd.DataFrame({'Open': history_df_filtered['Open'], 'High': history_df_filtered['High'], 'Low': history_df_filtered['Low'], 'Close': history_df_filtered['Close']})
stock_prices = pd.DataFrame({'date': history_df_filtered.index, 
                             'open': history_df_filtered['Open'], 
                             'close': history_df_filtered['Close'], 
                             'high': history_df_filtered['High'], 
                             'low': history_df_filtered['Low']}) 
  
ohlc = stock_prices.loc[:, ['date', 'open', 'high', 'low', 'close']] 
ohlc['date'] = pd.to_datetime(ohlc['date']) 
ohlc['date'] = ohlc['date'].apply(mpl_dates.date2num) 
ohlc = ohlc.astype(float) 
  
# Creating Subplots 
fig, ax = plt.subplots() 
  
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='blue', 
                 colordown='green', alpha=0.4) 
  
# Setting labels & titles 
ax.set_xlabel('Date') 
ax.set_ylabel('Price') 
fig.suptitle('Stock Prices of a week') 
  
# Formatting Date 
date_format = mpl_dates.DateFormatter('%d-%m-%Y') 
ax.xaxis.set_major_formatter(date_format) 
fig.autofmt_xdate() 
  
fig.tight_layout() 

#display candlestick chart
plt.show()