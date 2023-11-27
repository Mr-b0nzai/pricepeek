import sqlite3
import hashlib
import socket
import threading
import socket
import threading
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr

print("Server running...")

yf.pdr_override()

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
today = pd.to_datetime("today").strftime("%Y-%m-%d")

input_shape = [14]
random_state = 0

# login_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # login_server.bind(('198.199.86.77', 4567)) # uncomment this line to run on a public server
# login_server.bind(("localhost", 4567))
# login_server.listen()

prediction_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# prediction_server.bind(('198.199.86.77', 5000)) # uncomment this line to run on a public server
prediction_server.bind(("localhost", 5000))
prediction_server.listen()

conn = sqlite3.connect("users.db")
cur = conn.cursor()


# FOR TESTING PURPOSES ONLY
cur.execute(
    "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(255) NOT NULL, password VARCHAR(255) NOT NULL)"
)

username1, password1 = "admin", hashlib.sha256("admin".encode()).hexdigest()
username2, password2 = "john", hashlib.sha256("pass".encode()).hexdigest()
username3, password3 = "doe", hashlib.sha256("blah".encode()).hexdigest()
username4, password4 = "jojo", hashlib.sha256("midget".encode()).hexdigest()
cur.execute(
    "INSERT INTO users (username, password) VALUES (?, ?)", (username1, password1)
)
cur.execute(
    "INSERT INTO users (username, password) VALUES (?, ?)", (username2, password2)
)
cur.execute(
    "INSERT INTO users (username, password) VALUES (?, ?)", (username3, password3)
)
cur.execute(
    "INSERT INTO users (username, password) VALUES (?, ?)", (username4, password4)
)

conn.commit()


# def handle_client(c):
#     try:
#         username = c.recv(1024).decode()
#         password = c.recv(1024).decode()
#         password = hashlib.sha256(password.encode()).hexdigest()

#         conn = sqlite3.connect("users.db")
#         cur = conn.cursor()

#         cur.execute(
#             "SELECT * FROM users WHERE username=? AND password=?", (username, password)
#         )

#         if cur.fetchall():
#             c.send("Login successful".encode())
#             print(f"{username} logged in")
#         else:
#             c.send("Login failed".encode())
#             print(f"{username} failed to log in")
#     except ConnectionResetError:
#         print(f"Connection reset by {c.getpeername()}")

#     finally:
#         c.close()


def handle_prediction(c):
    global ticker, days
    while True:
        try:
            ticker = c.recv(1024).decode()
            if ticker == "exit":
                break

            days = c.recv(1024).decode()
            c.send("Working on predictions...".encode())
            print(f"Received {ticker} and {days}")
            prediction_low, prediction_high, prediction_close = run_prediction(ticker)
            print(prediction_low.iloc[0, 0])
            time.sleep(3)
            c.send(str(prediction_low.iloc[0, 0]).encode())
            print("sent low")
            c.send(str(prediction_high.iloc[0, 0]).encode())
            print("sent high")
            c.send(str(prediction_close.iloc[0, 0]).encode())
            print("sent close")
            c.close()
        except ConnectionResetError:
            print(f"Prediction connection reset by {c.getpeername()}")


# def handle_prediction(c):
#     global ticker, days
#     while True:
#         try:
#             ticker = c.recv(1024).decode()
#             if ticker == "exit":
#                 break

#             days = c.recv(1024).decode()
#             c.send("Working on predictions...".encode())
#             print(f"Received {ticker} and {days}")
#             prediction_low, prediction_high, prediction_close = run_prediction(ticker)
#             print(prediction_low.iloc[0, 0])
#             time.sleep(3)
#             c.send(str(prediction_low.iloc[0, 0]).encode())
#             time.sleep(1)
#             c.send(str(prediction_high.iloc[0, 0]).encode())
#             time.sleep(1)
#             c.send(str(prediction_close.iloc[0, 0]).encode())

#         except ConnectionResetError:
#             print(f"Prediction connection reset by {c.getpeername()}")

#     c.close()


btc_data = None
print(btc_data)


def download_dataset(ticker):
    global btc_data
    if btc_data is not None and "ticker" in btc_data and btc_data["ticker"] == ticker:
        return btc_data["data"]
    else:
        try:
            btc_data_live = pdr.get_data_yahoo(ticker, start="1986-01-01", end=today)
            btc_data = {"ticker": ticker, "data": btc_data_live.copy()}
            btc_data["data"].reset_index(inplace=True)
            btc_data["data"].set_index("Date", inplace=True)

            # check if checkbox is checked
            # check to see if dataset already has current date
            if btc_data["data"].index[-1] != today:
                # Create a new row with the prices for the 27th
                new_row = pd.DataFrame(
                    {
                        "Open": [0],
                        "High": [0],
                        "Low": [0],
                        "Close": [0],
                        "Adj Close": [0],
                        "Volume": [0],
                    },
                    index=[today],
                )
                # Append the new row to the existing DataFrame
                btc_data["data"] = pd.concat([btc_data["data"], new_row])

            if btc_data["data"].empty:
                print(
                    "Error downloading data. Please make sure the ticker is correct and try again."
                )
            else:
                print("Data downloaded successfully")
                return btc_data["data"]
        except:
            print("There was a problem downloading the data")


def score_model_history(model):
    model = model.history
    history_df = pd.DataFrame(model)

    history_df_filtered = history_df[history_df.index >= 150]
    max_val_loss = history_df_filtered["val_loss"].max()
    # Start the plot at epoch 5. You can change this to get a different view.
    # history_df.loc[:, ["loss", "val_loss"]].plot()
    print("Minimum Validation Loss: {}".format(history_df["val_loss"].min()))
    print("Max Validation Loss: {}".format(max_val_loss))
    print("Mean Validation Loss: {}".format(history_df_filtered["val_loss"].mean()))


def create_feature_columns(dataset, y_value):
    # Create the shifted lag series of prior trading period close values
    # for i in range(0, lags):
    #     dataset["Lag%s" % str(i+1)] = dataset[y_value].shift(i+1).pct_change()

    dataset["Prediction"] = dataset[y_value].shift(
        -int(days)
    )  # creating new column called 'prediction' that contains the close price of the next 3 days
    dataset["Daily Return"] = dataset[
        "Close"
    ].pct_change()  # creating new column called 'Daily Return' that contains the daily return of the stock
    dataset["volume_gap"] = dataset.Volume.pct_change()
    dataset["ho"] = dataset["High"] - dataset["Open"]
    dataset["lo"] = dataset["Low"] - dataset["Open"]
    dataset["oc"] = dataset.Open - dataset.Close
    dataset["hl"] = dataset.High - dataset.Low
    dataset["Dates"] = pd.to_datetime(dataset.index)  # convert to datetime column
    dataset["day_of_week"] = dataset["Dates"].dt.dayofweek
    dataset["day_of_month"] = dataset["Dates"].dt.day
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
    dataset.drop("Dates", axis=1, inplace=True)
    dataset.dropna(inplace=True)  # dropping NaN values

    return dataset


def prepare_dataset(dataset, y_value="Prediction"):
    # create X and y
    X = dataset.drop("Prediction", axis=1)
    X_fcast = X[
        -int(days) :
    ]  # creating the set that we will use to predict the future values
    X = X[: -int(days)]  # removing the last 3 rows
    y = dataset[y_value]  # <==== UPDATE THIS FOR PREDICTED OUTCOME
    y = y[: -int(days)]  # removing the last 3 rows

    # remove missing values and outliers
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]
    X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
    y = y[X.index]

    # splitting train/test
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    # create a scaler object
    scaler = StandardScaler()

    # fit the scaler on the training data
    scaler.fit(X_train)

    # transform the training and forecast data
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_fcast_scaled = scaler.transform(X_fcast)

    return X_train_scaled, X_valid_scaled, y_train, y_valid, X_fcast_scaled


def run_preparation(ticker):
    try:
        columns = ["Low", "High", "Close"]
        X_train_scaled = {}
        X_valid_scaled = {}
        X_fcast_scaled = {}
        y_train = {}
        y_valid = {}
        datasets = {}

        for col in columns:
            dataset = download_dataset(ticker).copy()
            dataset.index = download_dataset(ticker).index

            create_feature_columns(dataset, col)

            result = prepare_dataset(dataset, y_value=col)

            # transform the training and forecast data
            X_train_scaled[col] = result[0]
            X_valid_scaled[col] = result[1]
            X_fcast_scaled[col] = result[4]
            y_train[col] = result[2]
            y_valid[col] = result[3]
            datasets[col] = dataset

        print("Data prepared successfully")
        return (
            X_train_scaled,
            X_valid_scaled,
            y_train,
            y_valid,
            X_fcast_scaled,
            datasets,
        )
    except:
        print("There was a problem preparing the data")


early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=200,  # how many epochs to wait before stopping
    restore_best_weights=True,
)


def run_prediction(ticker):
    download_dataset(ticker)
    run_preparation(ticker)

    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=input_shape),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    prediction_low = run_model(
        model,
        run_preparation(ticker)[0]["Low"],
        run_preparation(ticker)[1]["Low"],
        run_preparation(ticker)[2]["Low"],
        run_preparation(ticker)[3]["Low"],
        run_preparation(ticker)[4]["Low"],
        "Low",
        run_preparation(ticker)[5]["Low"],
        "Low",
    )

    prediction_high = run_model(
        model,
        run_preparation(ticker)[0]["High"],
        run_preparation(ticker)[1]["High"],
        run_preparation(ticker)[2]["High"],
        run_preparation(ticker)[3]["High"],
        run_preparation(ticker)[4]["High"],
        "High",
        run_preparation(ticker)[5]["High"],
        "High",
    )

    prediction_close = run_model(
        model,
        run_preparation(ticker)[0]["Close"],
        run_preparation(ticker)[1]["Close"],
        run_preparation(ticker)[2]["Close"],
        run_preparation(ticker)[3]["Close"],
        run_preparation(ticker)[4]["Close"],
        "Close",
        run_preparation(ticker)[5]["Close"],
        "Close",
    )

    # concatenate the two dataframes into a single dataframe
    predictions = pd.concat([prediction_low, prediction_high, prediction_close], axis=1)
    print(predictions)

    # print(timeframe)
    # def display_labels():
    #     timeframe_label = customtkinter.CTkLabel(
    #         root,
    #         text=f"Predicting {timeframe}",
    #         font=("Helvetica", 24, "bold"),
    #     )
    #     timeframe_label.grid(row=0, column=1, sticky="nsew")
    #     low_label = customtkinter.CTkLabel(
    #         root,
    #         text=f"Low Price: {prediction_low.iloc[0,0]:.2f}",
    #         font=("Helvetica", 24, "bold"),
    #     )
    #     low_label.grid(row=1, column=1, sticky="nsew")

    #     high_label = customtkinter.CTkLabel(
    #         root,
    #         text=f"High Price: {prediction_high.iloc[0,0]:.2f}",
    #         font=("Helvetica", 24, "bold"),
    #     )
    #     high_label.grid(row=2, column=1, sticky="nsew")

    #     close_label = customtkinter.CTkLabel(
    #         root,
    #         text=f"Close Price: {prediction_close.iloc[0,0]:.2f}",
    #         font=("Helvetica", 24, "bold"),
    #     )
    #     close_label.grid(row=3, column=1, sticky="nsew")

    # display_labels()

    return prediction_low, prediction_high, prediction_close

    # Start a new thread to run the prediction

    # history_df = pd.DataFrame(btc_data)
    # history_df = download_dataset(my_entry.get()).copy()
    # history_df.index = pd.DatetimeIndex(history_df.index).astype(np.int64) // 10**9
    # history_df_filtered = history_df[history_df.index >= pd.Timestamp(150, unit='D').timestamp()]

    # create stock_prices DataFrame
    # last_7_days = history_df_filtered[-7:]
    # stock_prices = pd.DataFrame({'open': last_7_days['Open'],
    #                             'close': last_7_days['Close'],
    #                             'high': last_7_days['High'],
    #                             'low': last_7_days['Low'],
    #                             'volume': 0})

    # # add date column to DataFrame
    # last_7_dates = list(pd.DatetimeIndex(last_7_days.index).strftime('%Y-%m-%d'))
    # stock_prices['date'] = last_7_dates

    # # set date as index
    # stock_prices.set_index(pd.DatetimeIndex(stock_prices['date']), inplace=True)

    # # remove previous chart if present
    # for widget in frame.winfo_children():
    #     if isinstance(widget, FigureCanvasTkAgg):
    #         widget.get_tk_widget().destroy()

    # # create candlestick chart using mplfinance
    # fig, ax = mpf.plot(stock_prices, type='candlestick', style='charles', volume=True, figratio=(16,9), mav=(5,10,20), title='Last 7 Days of Stock Prices', returnfig=True)

    # # create canvas for chart
    # canvas = FigureCanvasTkAgg(fig, master=frame)
    # canvas.draw()
    # canvas.get_tk_widget().pack()

    # # save canvas for future removal
    # global chart_canvas
    # chart_canvas = canvas


def run_model(
    model,
    X_train_scaled,
    X_valid_scaled,
    y_train,
    y_valid,
    X_fcast_scaled,
    y_value,
    dataset,
    y_col,
):
    # def custom_loss(y_valid, y_pred):
    #     # check if y_pred has only one column
    #     if y_pred.shape[1] == 1:
    #         # if so, add a column of zeros to y_pred
    #         y_pred = K.concatenate([y_pred, K.zeros_like(y_pred)], axis=1)
    #     # calculate the difference between predicted Close and High prices
    #     diff = y_pred[:, 1] - y_pred[:, 0]
    #     # apply a penalty if the difference is negative (i.e., predicted Close price is higher than predicted High price)
    #     penalty = K.mean(K.maximum(0.0, -diff))
    #     # calculate the mean absolute error between predicted and true prices
    #     mae = K.mean(K.abs(y_valid - y_pred))
    #     # add the penalty to the mean absolute error
    #     loss = mae + penalty
    #     return loss

    model.compile(
        optimizer="adam",
        loss="mae",
    )

    y_model = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_valid_scaled, y_valid),
        batch_size=256,
        epochs=500,
        callbacks=[early_stopping],
        verbose=0,  # hide the output because we have so many epochs
    )

    forcast = model.predict(X_fcast_scaled)
    print(score_model_history(y_model))
    forcast = forcast[-int(days) :]  # select only the last 5 predictions
    forcast = np.reshape(
        forcast, (forcast.shape[0], forcast.shape[1])
    )  # reshape into a 5x2 array

    # create dataframe for Low predictions
    d = dataset[[y_value]].tail(len(forcast))
    d = d.tail(int(days))
    d.index = pd.date_range(
        start=download_dataset(ticker).index[-1],
        periods=(len(d)),
        freq="D",
    )
    prediction = pd.DataFrame(forcast)
    prediction.index = d.index
    # prediction['Actual Price'] = download_dataset(my_entry.get())[y_value]
    prediction.rename(columns={0: "Forecasted " + y_col + " Price"}, inplace=True)

    return prediction


while True:
    # c, addr = login_server.accept()
    # print(f"Connected to {addr}")
    # t = threading.Thread(target=handle_client, args=(c,))
    # t.start()

    pred_c, pred_addr = prediction_server.accept()
    print(f"Prediction server connected to {pred_addr}")
    t1 = threading.Thread(target=handle_prediction, args=(pred_c,))
    t1.start()

    # pred_conn, pred_addr = prediction_server.accept()
    # print(f"Connected to {pred_addr}")
    # t1 = threading.Thread(target=handle_prediction, args=(pred_conn,))
    # t1.start()
