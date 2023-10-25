from tkinter import *
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
# from stock_indicators import indicators, Quote
import sys
# from clr_loader import get_coreclr
# from pythonnet import set_runtime

# rt = get_coreclr(r"C:\Users\ninja\AppData\Local\Programs\Python\Python311\Lib\site-packages\pythonnet\runtime\Python.Runtime.deps.json")
# set_runtime(rt)

# import clr

yf.pdr_override()

from datetime import datetime

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

future_pred = 2 # Number of days to forcast
# lag featuers
lags = 1
input_shape = [15]
random_state = 0

root = Tk()
root.geometry("500x500")

frame = Frame(root)
frame.pack()

root.title("PricePeek")

my_entry = Entry(frame, width = 20)
my_entry.insert(0,'Type Ticker Here')
my_entry.pack(padx = 5, pady = 5)

var = StringVar()
var.set("Welcome to PricePeek")

btc_data = None

def download_dataset():
    global btc_data
    if btc_data is not None:
        return btc_data
    else:
        try:
            btc_data_live = pdr.get_data_yahoo(my_entry.get(), start='1986-01-01', end=end)
            # btc_data = pd.read_csv('BTC-USD-MAX.csv', dtype=dtype, parse_dates=['Date'])
            btc_data = btc_data_live.copy()
            btc_data.reset_index(inplace=True)
            btc_data.set_index('Date', inplace=True)
            if (btc_data.empty):
                var.set("Error downloading data. Please make sure the ticker is correct and try again.")
            else:
                var.set("Data downloaded successfully")
                return btc_data
        except:
            var.set("There was a problem downloading the data")


def score_model_history(model):
    model = model.history
    history_df = pd.DataFrame(model)

    history_df_filtered = history_df[history_df.index >= 150]
    max_val_loss = history_df_filtered['val_loss'].max()
    # Start the plot at epoch 5. You can change this to get a different view.
    history_df.loc[:, ['loss', 'val_loss']].plot()
    print("Minimum Validation Loss: {}".format(history_df['val_loss'].min()))
    print("Max Validation Loss: {}".format(max_val_loss))
    print("Mean Validation Loss: {}".format(history_df_filtered['val_loss'].mean()))

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
    # quotes_list = [
    #     Quote(d,o,h,l,c,v)
    #     for d,o,h,l,c,v
    #     in zip(dataset['Dates'], dataset['Open'], dataset['High'], dataset['Low'], dataset['Close'], dataset['Volume'])
    # ]

    # results = indicators.get_macd(quotes_list)

    # for r in results:
    #     dataset.loc[dataset['Dates'] == r.date, 'MACD'] = r.macd
        
    # vol_ma = indicators.get_ema(quotes_list, 5)

    # for r in vol_ma:
    #     dataset.loc[dataset['Dates'] == r.date, 'EMA'] = r.ema

    # dataset['MACD'].fillna(0, inplace=True)

    # dataset['EMA'].fillna(0, inplace=True)
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

def run_preparation():
    try:
        # HIGH PREDICTIONS
        dataset_high = download_dataset().copy()
        dataset_high.index = download_dataset().index

        create_feature_columns(dataset_high, 'High')

        high_result = prepare_dataset(dataset_high)

        # transform the training and forecast data
        X_train_high_scaled = high_result[0]
        X_valid_high_scaled = high_result[1]
        X_fcast_high_scaled = high_result[4]
        y_train_high = high_result[2]
        y_valid_high = high_result[3]

        # LOW PREDICTIONS
        dataset_low = download_dataset().copy()
        dataset_low.index = download_dataset().index

        create_feature_columns(dataset_low, 'Low')

        low_result = prepare_dataset(dataset_low)

        # transform the training and forecast data
        X_train_low_scaled = low_result[0]
        X_valid_low_scaled = low_result[1]
        X_fcast_low_scaled = low_result[4]
        y_train_low = low_result[2]
        y_valid_low = low_result[3]

        # Close PREDICTIONS
        dataset_close = download_dataset().copy()
        dataset_close.index = download_dataset().index

        create_feature_columns(dataset_close, 'Close')

        close_result = prepare_dataset(dataset_close)

        # transform the training and forecast data
        X_train_close_scaled = close_result[0]
        X_valid_close_scaled = close_result[1]
        X_fcast_close_scaled = close_result[4]
        y_train_close = close_result[2]
        y_valid_close = close_result[3]
        var.set('Data prepared successfully')
        return X_train_close_scaled, X_valid_close_scaled, y_train_close, y_valid_close, X_fcast_close_scaled, dataset_close
    except:
        var.set("There was a problem preparing the data")
        
early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=200, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

def run_prediction():
    
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
    
    # prediction_low = run_model(model, run_preparation()[0], X_valid_low_scaled, y_train_low, y_valid_low, X_fcast_low_scaled, 'Low', dataset_low)
    # prediction_high = run_model(model, X_train_high_scaled, X_valid_high_scaled, y_train_high, y_valid_high, X_fcast_high_scaled, 'High', dataset_high)
    prediction_close = run_model(model, run_preparation()[0], run_preparation()[1], run_preparation()[2], run_preparation()[3], run_preparation()[4], 'Close', run_preparation()[5])

    # concatenate the two dataframes into a single dataframe
    predictions = pd.concat([prediction_close], axis=1)
    print(predictions)
    prediction_label = Label(frame, text = predictions)
    prediction_label.pack()
    return predictions


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
    d.index = pd.date_range(start=download_dataset().index[-1], periods=(len(d)), freq='D')
    prediction = pd.DataFrame(forcast)
    prediction.index = d.index
    prediction['Actual Price'] = download_dataset()[y_value]
    prediction.rename(columns = {0: 'Forecasted Price'}, inplace=True)

    return prediction

label = Label(frame, textvariable = var )
label.pack()
download_btn = Button(frame, text = "Download", command = download_dataset)
download_btn.pack()

prepare_btn = Button(frame, text = "Prepare Data", command = run_preparation)
prepare_btn.pack()

run_btn = Button(frame, text = "Run Prediction", command = run_prediction)
run_btn.pack()

root.mainloop()