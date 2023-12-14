import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tpqoa
import sys

# Load OANDA config file
api = tpqoa.tpqoa('oanda.cfg')

# Getting data
df = api.get_history("EUR_USD", "2017-01-01", "2023-11-30", "M30", "M")
df.to_csv('./data/EUR_USD_M30_2017_2023_C.csv', encoding='utf-8')

# Import data
data = pd.read_csv('./data/EUR_USD_M30_2017_2023_C.csv', 
                   parse_dates = ['time'], index_col = 'time', 
                   usecols = ['time', 'c'])
data.rename(columns = {'c' : 'EUR_USD'}, inplace = True)
symbol = data.columns[0]

# Calculate log returns
data['returns'] = np.log(data[symbol] / data[symbol].shift())

# # Visualize 
# data.plot(figsize = (12, 8))
# plt.title('EUR_USD_M30')
# plt.savefig('./plot/EUR_USD_M30.png')

# Calculate log returns
data['returns'] = np.log(data[symbol] / data[symbol].shift())
print(data)

# Adding features
window = 48

df = data.copy()

# SMA"", 
df["dir"] = np.where(df["returns"] > 0, 1, 0)
df["sma200"] = df[symbol].rolling(window).mean() - df[symbol].rolling(200).mean()
# df["sma5_8"] = df[symbol].rolling(sma5).mean() - df[symbol].rolling(sma8).mean()
# df["sma8_13"] = df[symbol].rolling(sma8).mean() - df[symbol].rolling(sma13).mean()

# EMA 5-8-13
# df["ema"] = df[symbol].ewm(span=window, adjust=False).mean()
df["ema5_8"] = df[symbol].ewm(5, adjust=False).mean() - df[symbol].ewm(8, adjust=False).mean()
df["ema8_13"] = df[symbol].ewm(8, adjust=False).mean() - df[symbol].ewm(13, adjust=False).mean()

# Bollinger band
df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window).std()

# RSI_14
# Calculate the average gain and average loss
df['Gain'] = np.where(df['returns'] > 0, df['returns'], 0)
df['Loss'] = np.where(df['returns'] < 0, -df['returns'], 0)

df['AvgGain'] = df['Gain'].rolling(14).mean()
df['AvgLoss'] = df['Loss'].rolling(14).mean()
# Calculate the relative strength (RS)
df['RS'] = df['AvgGain'] / df['AvgLoss']

# Calculate the RSI
df['RSI'] = 100 - (100 / (1 + df['RS']))

df['rsi'] = 0
# Buy Signal: RSI crosses below 30
df.loc[df['RSI'] < 30, 'rsi'] = 1

# Sell Signal: RSI crosses above 70
df.loc[df['RSI'] > 70, 'rsi'] = -1

# Momentum
df["mom"] = df["returns"].rolling(3).mean()

# Volatile
df["vol"] = df["returns"].rolling(window).std()
df.dropna(inplace = True)
# print(df)

# Adding lags to features
lags = 5
cols = []
features = ["dir", "sma200", "ema5_8", "ema8_13", "boll", "rsi", "mom", "vol"]

for f in features:
  for lag in range(1, lags + 1):
    col = "{}_lag_{}".format(f, lag)
    df[col] = df[f].shift(lag)
    cols.append(col)
df.dropna(inplace = True)
# print(df)

# Prepare train and test
# print(len(df))
# 85833

# Split 2/3 to train and 1/3 to test
split = int(len(df) * 2 / 3)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# print(train[cols])

# Data normalize / Feature Scaling
mu, std = train.mean(), train.std()
train_s = (train - mu) / std
# print(train_s)
# train_s.describe()
# print(train_s.describe())

# Model 
from DNNModel import *

# # Train
# set_seeds(100)
# model = create_model(hl = 3, hu = 50, dropout = True, input_dim = len(cols))
# model.fit(x = train_s[cols], y = train["dir"], epochs = 50, verbose = False,
#           validation_split = 0.2, shuffle = False, class_weight = cw(train))
# model.evaluate(train_s[cols], train["dir"])

# pred = model.predict(train_s[cols]) # prediction (probabilities)

# # Test
# test_s = (test - mu) / std
# model.evaluate(test_s[cols], test["dir"])
# pred = model.predict(test_s[cols])

# test["proba"] = model.predict(test_s[cols])
# test["position"] = np.where(test.proba < 0.47, -1, np.nan) # 1. short where proba < 0.47
# test["position"] = np.where(test.proba > 0.53, 1, test.position) # 2. long where proba > 0.53
# test.index = test.index.tz_localize("UTC")
# test["NYTime"] = test.index.tz_convert("America/New_York")
# test["hour"] = test.NYTime.dt.hour
# test["position"] = np.where(~test.hour.between(2, 12), 0, test.position) # 3. neutral in non-busy hours
# test["position"] = test.position.ffill().fillna(0) # 4. in all other cases: hold position
# test.position.value_counts(dropna = False)

# test["strategy"] = test["position"] * test["returns"]
# test["creturns"] = test["returns"].cumsum().apply(np.exp)
# test["cstrategy"] = test["strategy"].cumsum().apply(np.exp)

# # Saving Model and Parameters
# model.save("DNN_model")

# Loading mu and std
import pickle
params = {"mu":mu, "std":std}
pickle.dump(params, open("params.pkl", "wb"))
# params = pickle.load(open("params.pkl", "rb"))
# mu = params["mu"]
# std = params["std"]

# Implementation with OANDA
from datetime import datetime, timedelta
import keras
import time

model = keras.models.load_model("DNN_model")
params = pickle.load(open("params.pkl", "rb"))
mu = params["mu"]
std = params["std"]

class DNNTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, lags, model, mu, std, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []

        #*****************add strategy-specific attributes here******************
        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        #************************************************************************

    def get_most_recent(self, days = 5):
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                self.start_time = pd.to_datetime(datetime.utcnow()).tz_localize("UTC") # NEW -> Start Time of Trading Session
                break

    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ", flush = True)

        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2},
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()

    def resample_and_join(self):
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length,
                                                                          label="right").last().ffill().iloc[:-1]])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]

    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()

        #******************** define your strategy here ************************
        #create features
        # df = df.append(self.tick_data) # append latest tick (== open price of current bar)
        # df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        # df["dir"] = np.where(df["returns"] > 0, 1, 0)
        # df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        # df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        # df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        # df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        # df["mom"] = df["returns"].rolling(3).mean()
        # df["vol"] = df["returns"].rolling(self.window).std()
        # df.dropna(inplace = True)

        # df = df.append(self.tick_data) # append latest tick (== open price of current bar)
        df = pd.concat([df, self.tick_data], ignore_index=True)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, 0)
        df["sma200"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(200).mean()
        df["ema5_8"] = df[self.instrument].ewm(5, adjust=False).mean() - df[self.instrument].ewm(8, adjust=False).mean()
        df["ema8_13"] = df[self.instrument].ewm(8, adjust=False).mean() - df[self.instrument].ewm(13, adjust=False).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        # df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        # df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        # RSI_14
        # Calculate the average gain and average loss
        df['Gain'] = np.where(df['returns'] > 0, df['returns'], 0)
        df['Loss'] = np.where(df['returns'] < 0, -df['returns'], 0)

        df['AvgGain'] = df['Gain'].rolling(14).mean()
        df['AvgLoss'] = df['Loss'].rolling(14).mean()
        # Calculate the relative strength (RS)
        df['RS'] = df['AvgGain'] / df['AvgLoss']

        # Calculate the RSI
        df['RSI'] = 100 - (100 / (1 + df['RS']))

        df['rsi'] = 0
        # Buy Signal: RSI crosses below 30
        df.loc[df['RSI'] < 30, 'rsi'] = 1

        # Sell Signal: RSI crosses above 70
        df.loc[df['RSI'] > 70, 'rsi'] = -1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)

        # create lags
        self.cols = []
        # features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
        features = ["dir", "sma200", "ema5_8", "ema8_13", "boll", "rsi", "mom", "vol"]


        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace = True)

        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.model.predict(df_s[self.cols])

        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position"] = np.where(df.proba < 0.47, -1, np.nan)
        df["position"] = np.where(df.proba > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************

        self.data = df.copy()

    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0

    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

trader = DNNTrader("oanda.cfg", "EUR_USD", bar_length = "30min",
                   window = 48, lags = 5, model = model, mu = mu, std = std, units = 100000)

trader.get_most_recent()
trader.stream_data(trader.instrument, stop = 10000)
if trader.position != 0:
    close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units,
                                      suppress = True, ret = True)
    trader.report_trade(close_order, "GOING NEUTRAL")
    trader.position = 0

trade_result = trader.data
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"output_{current_time}.csv"
trade_result.to_csv('./result/output.csv', index=False)