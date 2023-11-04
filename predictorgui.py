print("Starting program...")

import threading
from tkinter import *
import tkinter
from tkinter import ttk
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# import tensorflow.keras.backend as K
import yfinance as yf
from pandas_datareader import data as pdr
import customtkinter
from datetime import datetime
import sys
import loginfunctions as login
from PIL import Image, ImageTk

# from stock_indicators import indicators, Quote
# from clr_loader import get_coreclr
# from pythonnet import set_runtime

# rt = get_coreclr(r"C:\Users\ninja\AppData\Local\Programs\Python\Python311\Lib\site-packages\pythonnet\runtime\Python.Runtime.deps.json")
# set_runtime(rt)

# import clr

yf.pdr_override()


def on_closing():
    # add any cleanup code here
    root.destroy()
    sys.exit()


end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
today = pd.to_datetime("today").strftime("%Y-%m-%d")

is_logged_in = login.is_logged_in
print(is_logged_in)
future_pred = 1  # Default number of days to forecast
# lag featuers
# lags = 1
input_shape = [14]
random_state = 0

root = customtkinter.CTk()
w = 1100  # width for the Tk root
h = 500  # height for the Tk root

# get screen width and height
ws = root.winfo_screenwidth()  # width of the screen
hs = root.winfo_screenheight()  # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)

# set the dimensions of the screen
# and where it is placed
root.geometry("%dx%d+%d+%d" % (w, h, x, y))
root.title("PricePeek")
# add system path to icon
root.iconbitmap(
    "H:\\Users\\josiah\\Documents\\pricepeek\\assets\\logo-pricepeek-mark.ico",
    default="H:\\Users\\josiah\\Documents\\pricepeek\\assets\\logo-pricepeek-mark.ico",
)

root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure((2, 3), weight=0)
root.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)
root.grid_rowconfigure(3, weight=0)
root.grid_rowconfigure(4, weight=0)
root.grid_rowconfigure(5, weight=0)

var = StringVar()
var.set("Welcome to PricePeek")

# customtkinter.set_appearance_mode('system')
customtkinter.set_default_color_theme("dark-blue")


def open_input_dialog_event():
    dialog = customtkinter.CTkInputDialog(
        text="Type in a number:", title="CTkInputDialog"
    )
    print("CTkInputDialog:", dialog.get_input())


def change_appearance_mode_event(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)


def change_scaling_event(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    customtkinter.set_widget_scaling(new_scaling_float)


def sidebar_button_event(root):
    print("sidebar_button click")


login_window_shown = False
for i in range(1):
    # add PeekPrice logo
    logo_path = Image.open(
        "H:\\Users\\josiah\\Documents\\pricepeek\\assets\\logo-pricepeek-mark.png"
    )
    logo_img = ImageTk.PhotoImage(logo_path)

    def show_login_success(username, password, login_window):
        global is_logged_in
        result = login.send_credentials(username, password)
        is_logged_in = result
        # print(is_logged_in)
        if result == True:
            login_window.destroy()
            root.deiconify()
        elif result == False and is_logged_in == False:
            if not hasattr(login_window, "login_success"):
                print("Creating new login_success label")
                login_success = customtkinter.CTkLabel(
                    master=login_frame, text="Login failed!", font=("Arial Bold", 20)
                )
                login_success.pack()
                login_window.login_success = login_success
            else:
                print("Updating existing login_success label")
                login_window.login_success.configure(text="Login failed!")

    root.withdraw()

    login_window = customtkinter.CTkToplevel()
    login_frame = customtkinter.CTkFrame(master=login_window, bg_color="transparent")
    login_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
    login_window.geometry("%dx%d+%d+%d" % (600, 400, x, y))
    login_window.wm_iconbitmap()
    login_window.after(300, lambda: login_window.iconphoto(False, logo_img))
    login_window.title("Login")

    login_frame.grid_rowconfigure(1, weight=1)
    login_frame.grid_columnconfigure(1, weight=1)

    # add PeekPrice logo
    logo = Image.open("assets/logo-pricepeek-white-medium.png")
    login_img = ImageTk.PhotoImage(logo)
    logo_label = customtkinter.CTkLabel(master=login_frame, text="", image=login_img)
    logo_label.pack()

    username = customtkinter.CTkLabel(
        master=login_frame, text="Username", font=("Arial Bold", 24)
    )
    username.pack()
    username_entry = customtkinter.CTkEntry(login_frame, placeholder_text="username")
    username_entry.pack(pady=10)

    password = customtkinter.CTkLabel(
        master=login_frame, text="Password", font=("Arial Bold", 24)
    )
    password.pack()
    password_entry = customtkinter.CTkEntry(
        login_frame, placeholder_text="password", show="*"
    )
    password_entry.pack(pady=10)

    login_btn = customtkinter.CTkButton(
        master=login_frame,
        text="Login",
        command=lambda: show_login_success(
            username_entry.get(), password_entry.get(), login_window
        ),
    )
    login_btn.pack(pady=10)

    login_window.protocol("WM_DELETE_WINDOW", on_closing)


# create sidebar frame with widgets
sidebar_frame = customtkinter.CTkFrame(master=root, width=140, corner_radius=0)
sidebar_frame.grid(row=0, column=0, rowspan=8, sticky="nsew")
sidebar_frame.grid_rowconfigure(8, weight=1)
# add PeekPrice logo
logo_image = Image.open("assets/logo-pricepeek-white-medium.png")
login_img = ImageTk.PhotoImage(logo_image)
logo_label = customtkinter.CTkLabel(sidebar_frame, text="", image=login_img)
logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
sidebar_button_1 = customtkinter.CTkButton(
    sidebar_frame, text="How to Use", command=sidebar_button_event
)
sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
sidebar_button_2 = customtkinter.CTkButton(
    sidebar_frame, text="Saved Predictions", command=sidebar_button_event
)
sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
sidebar_button_3 = customtkinter.CTkButton(
    sidebar_frame, text="Premium Feature", command=sidebar_button_event
)
sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
appearance_mode_label = customtkinter.CTkLabel(
    sidebar_frame, text="Appearance Mode:", anchor="w"
)
appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
appearance_mode_optionemenu = customtkinter.CTkOptionMenu(
    sidebar_frame,
    values=["Light", "Dark", "System"],
    command=change_appearance_mode_event,
)
appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
scaling_label = customtkinter.CTkLabel(sidebar_frame, text="UI Scaling:", anchor="w")
scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
scaling_optionemenu = customtkinter.CTkOptionMenu(
    sidebar_frame,
    values=["80%", "90%", "100%", "110%", "120%"],
    command=change_scaling_event,
)
scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

# label = customtkinter.CTkLabel(master=frame, textvariable = var , font=("Arial Bold", 15))
# label.pack()

# pred_label = customtkinter.CTkLabel(frame, text="Enter number of days to predict:")
# pred_label.pack()

# # Create an Entry widget for the user to input a value
# future_pred_entry = customtkinter.CTkEntry(frame, placeholder_text='Days to predict')
# future_pred_entry.pack()

# create radiobutton frame
radiobutton_frame = customtkinter.CTkFrame(root)
radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
radio_var = tkinter.IntVar(value=0)
label_radio_group = customtkinter.CTkLabel(
    master=radiobutton_frame, text="Prediction Period:"
)
label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")
radio_button_1 = customtkinter.CTkRadioButton(
    master=radiobutton_frame, text="End of Today", variable=radio_var, value=1
)
radio_button_1.grid(row=1, column=2, pady=10, padx=20, sticky="n")
radio_button_2 = customtkinter.CTkRadioButton(
    master=radiobutton_frame, text="Today + Tomorrow", variable=radio_var, value=2
)
radio_button_2.grid(row=2, column=2, pady=10, padx=20, sticky="n")
radio_button_3 = customtkinter.CTkRadioButton(
    master=radiobutton_frame, text="3 Days (Coming soon)", variable=radio_var, value=3
)
radio_button_3.grid(row=3, column=2, pady=10, padx=20, sticky="n")
print(radio_var.get())

my_entry = customtkinter.CTkEntry(root, placeholder_text="Ticker symbol")
# my_entry.insert(0)
my_entry.grid(row=5, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

btc_data = None

# def make_uppercase():
#     # Retrieve the text from the Entry widget and convert it to all capital letters
#     text = my_entry.get().upper()
#     # Update the text in the Entry widget
#     my_entry.delete(0, END)
#     my_entry.insert(0, text)

# print (make_uppercase())


def download_dataset(ticker):
    global btc_data
    if btc_data is not None and "ticker" in btc_data and btc_data["ticker"] == ticker:
        return btc_data["data"]
    else:
        try:
            btc_data_live = pdr.get_data_yahoo(ticker, start="1986-01-01", end=end)
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
                var.set(
                    "Error downloading data. Please make sure the ticker is correct and try again."
                )
            else:
                var.set("Data downloaded successfully")
                return btc_data["data"]
        except:
            var.set("There was a problem downloading the data")


def score_model_history(model):
    model = model.history
    history_df = pd.DataFrame(model)

    history_df_filtered = history_df[history_df.index >= 150]
    max_val_loss = history_df_filtered["val_loss"].max()
    # Start the plot at epoch 5. You can change this to get a different view.
    history_df.loc[:, ["loss", "val_loss"]].plot()
    print("Minimum Validation Loss: {}".format(history_df["val_loss"].min()))
    print("Max Validation Loss: {}".format(max_val_loss))
    print("Mean Validation Loss: {}".format(history_df_filtered["val_loss"].mean()))


def create_feature_columns(dataset, y_value):
    # Create the shifted lag series of prior trading period close values
    # for i in range(0, lags):
    #     dataset["Lag%s" % str(i+1)] = dataset[y_value].shift(i+1).pct_change()

    dataset["Prediction"] = dataset[y_value].shift(
        -int(radio_var.get())
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
        -int(radio_var.get()) :
    ]  # creating the set that we will use to predict the future values
    X = X[: -int(radio_var.get())]  # removing the last 3 rows
    y = dataset[y_value]  # <==== UPDATE THIS FOR PREDICTED OUTCOME
    y = y[: -int(radio_var.get())]  # removing the last 3 rows

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


def run_preparation():
    try:
        columns = ["Low", "High", "Close"]
        X_train_scaled = {}
        X_valid_scaled = {}
        X_fcast_scaled = {}
        y_train = {}
        y_valid = {}
        datasets = {}

        for col in columns:
            dataset = download_dataset(my_entry.get()).copy()
            dataset.index = download_dataset(my_entry.get()).index

            create_feature_columns(dataset, col)

            result = prepare_dataset(dataset, y_value=col)

            # transform the training and forecast data
            X_train_scaled[col] = result[0]
            X_valid_scaled[col] = result[1]
            X_fcast_scaled[col] = result[4]
            y_train[col] = result[2]
            y_valid[col] = result[3]
            datasets[col] = dataset

        var.set("Data prepared successfully")
        return (
            X_train_scaled,
            X_valid_scaled,
            y_train,
            y_valid,
            X_fcast_scaled,
            datasets,
        )
    except:
        var.set("There was a problem preparing the data")


early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=200,  # how many epochs to wait before stopping
    restore_best_weights=True,
)


def run_prediction():
    def run_prediction_thread():
        run_btn.configure(state="disabled")
        # Create a spinning wheel widget
        spinner = customtkinter.CTkProgressBar(root, mode="indeterminate")
        spinner.grid(row=4, column=1, columnspan=2, padx=(20, 0), sticky="nsew")
        spinner.start()
        download_dataset(my_entry.get())
        run_preparation()

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
            run_preparation()[0]["Low"],
            run_preparation()[1]["Low"],
            run_preparation()[2]["Low"],
            run_preparation()[3]["Low"],
            run_preparation()[4]["Low"],
            "Low",
            run_preparation()[5]["Low"],
            "Low",
        )

        prediction_high = run_model(
            model,
            run_preparation()[0]["High"],
            run_preparation()[1]["High"],
            run_preparation()[2]["High"],
            run_preparation()[3]["High"],
            run_preparation()[4]["High"],
            "High",
            run_preparation()[5]["High"],
            "High",
        )

        prediction_close = run_model(
            model,
            run_preparation()[0]["Close"],
            run_preparation()[1]["Close"],
            run_preparation()[2]["Close"],
            run_preparation()[3]["Close"],
            run_preparation()[4]["Close"],
            "Close",
            run_preparation()[5]["Close"],
            "Close",
        )

        # concatenate the two dataframes into a single dataframe
        predictions = pd.concat(
            [prediction_low, prediction_high, prediction_close], axis=1
        )
        print(predictions)

        if radio_var.get() == 1:
            timeframe = "End of today"
        elif radio_var.get() == 2:
            timeframe = "Today + tomorrow"
        elif radio_var.get() == 3:
            timeframe = "3 days"

        def display_labels():
            timeframe_label = customtkinter.CTkLabel(
                root,
                text=f"Predicting {timeframe}",
                font=("Helvetica", 24, "bold"),
            )
            timeframe_label.grid(row=0, column=1, sticky="nsew")
            low_label = customtkinter.CTkLabel(
                root,
                text=f"Low Price: {prediction_low.iloc[0,0]:.2f}",
                font=("Helvetica", 24, "bold"),
            )
            low_label.grid(row=1, column=1, sticky="nsew")

            high_label = customtkinter.CTkLabel(
                root,
                text=f"High Price: {prediction_high.iloc[0,0]:.2f}",
                font=("Helvetica", 24, "bold"),
            )
            high_label.grid(row=2, column=1, sticky="nsew")

            close_label = customtkinter.CTkLabel(
                root,
                text=f"Close Price: {prediction_close.iloc[0,0]:.2f}",
                font=("Helvetica", 24, "bold"),
            )
            close_label.grid(row=3, column=1, sticky="nsew")

        display_labels()

        # Stop the spinning wheel
        spinner.stop()
        spinner.destroy()
        run_btn.configure(state="normal")
        return predictions

    # Start a new thread to run the prediction
    prediction_thread = threading.Thread(target=run_prediction_thread)
    prediction_thread.start()

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
    forcast = forcast[-int(radio_var.get()) :]  # select only the last 5 predictions
    forcast = np.reshape(
        forcast, (forcast.shape[0], forcast.shape[1])
    )  # reshape into a 5x2 array

    # create dataframe for Low predictions
    d = dataset[[y_value]].tail(len(forcast))
    d = d.tail(int(radio_var.get()))
    d.index = pd.date_range(
        start=download_dataset(my_entry.get()).index[-1], periods=(len(d)), freq="D"
    )
    prediction = pd.DataFrame(forcast)
    prediction.index = d.index
    # prediction['Actual Price'] = download_dataset(my_entry.get())[y_value]
    prediction.rename(columns={0: "Forecasted " + y_col + " Price"}, inplace=True)

    return prediction


# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import mplfinance as mpf


# download_btn = Button(frame, text = "Download", command = download_dataset)
# download_btn.pack()

# prepare_btn = Button(frame, text = "Prepare Data", command = run_preparation)
# prepare_btn.pack()

# set default values
sidebar_button_2.configure(state="disabled", text="Saved Predictions")
sidebar_button_3.configure(state="disabled", text="Give Feedback")
radio_button_1.select()
radio_button_3.configure(state="disabled")
appearance_mode_optionemenu.set("System")
scaling_optionemenu.set("100%")

run_btn = customtkinter.CTkButton(
    master=root, text="Run Prediction", command=run_prediction
)
run_btn.grid(row=5, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

root.configure(background="#060e2e")


root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
