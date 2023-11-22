import re
from datetime import datetime
import os
import pickle
from datetime import timedelta

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
import requests
from sklearn.preprocessing import StandardScaler

from tradeTesting.CryptoPortfolio import CryptoPortfolio

def percent_change(old_value, new_value):
  return ((new_value - old_value) / old_value) * 100

def getXWithWindowTransformed(klines, scaler_x, window_size):
    klines = klines[-1*window_size:]
    opens = [float(kline["open"]) for kline in klines]
    highs = [float(kline["high"]) for kline in klines]
    lows = [float(kline["low"]) for kline in klines]
    closes = [float(kline["close"]) for kline in klines]
    volumes = [float(kline["volume"]) for kline in klines]
    X = np.hstack((opens, highs, lows,
                            closes, volumes))

    X = scaler_x.transform([X])


    return X

def getAllDataWithWindowTransformed(klines, scaler_x, window_size):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    opens = [float(kline["open"]) for kline in klines]
    highs = [float(kline["high"]) for kline in klines]
    lows = [float(kline["low"]) for kline in klines]
    closes = [float(kline["close"]) for kline in klines]
    volumes = [float(kline["volume"]) for kline in klines]
    X = []
    Y = []
    for i in range(len(closes) - window_size):
        # Stack the lists horizontally
        X.append(np.hstack((opens[i:i + window_size], highs[i:i + window_size], lows[i:i + window_size],
                            closes[i:i + window_size], volumes[i:i + window_size])))
        Y.append(closes[i + window_size])

    Y = np.array(Y)
    Y = Y.reshape(-1, 1)
    scaler_x.fit(X)
    scaler_y.fit(Y)
    X = scaler_x.transform(X)
    Y = scaler_y.transform(Y)

    return X,Y

symbols = ["SOLBUSD"]
invest_in_usd_each_symbol = 100
retrain_model = False
# Connect to the MongoDB database
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["crypto"]
model_collection = db["models"]

symbols = model_collection.distinct("symbol")
pattern = '(BUSD|USDT)$'
symbols = [s for s in symbols if re.search(pattern, s)]
symbols_count = {}
for symbol in symbols:
    base_symbol = re.sub(pattern, '', symbol)
    if base_symbol in symbols_count:
        symbols_count[base_symbol].append(symbol)
    else:
        symbols_count[base_symbol] = [symbol]
# filter duplicate symbols
symbols = [symbols_count[base_symbol][0] for base_symbol in symbols_count if len(symbols_count[base_symbol]) > 0]


# Create a new GridFS bucket for storing the models
model_bucket = gridfs.GridFSBucket(db, "models")
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
trade_report_dir = f"TradeReports/{current_time}"
os.makedirs(trade_report_dir,exist_ok=True)
predicted_prices = []
actual_prices = []
interval = "1d"
total_protfolio = CryptoPortfolio()
# Loop over the symbols
for symbol in symbols:
    # Query the collection for the specified symbol and interval
    doc = model_collection.find_one({"symbol": symbol, "interval": interval})
    score = doc.get("score")
    if doc and score > 0.986:
        # Retrieve the model from the GridFS bucket
        model_id = doc.get("model_id") or doc.get("model")
        klines = []
        indicator_settings = None
        try:
            window_size = doc["window_size"]
        except:
            pass
        try:
            # indicator_settings= doc["indicator_settings"]
            collection = db[interval]
            klines = collection.find({"symbol": symbol})
            klines = list(klines)

            # Extract the timestamp, open, high, low, and close prices from the documents
            timestamps = [kline["timestamp"] for kline in klines]
            opens = [float(kline["open"]) for kline in klines]
            highs = [float(kline["high"]) for kline in klines]
            lows = [float(kline["low"]) for kline in klines]
            closes = [float(kline["close"]) for kline in klines]
            volumes = [float(kline["volume"]) for kline in klines]
        except:
            pass
        scaler_x = None
        scaler_y = None
        try:
            scaler_x = doc["scaler_x"]
            scaler_y = doc["scaler_y"]
            if scaler_x:


                scaler_x = pickle.loads(scaler_x)
                scaler_y = pickle.loads(scaler_y)
        except:
            continue
        try:
            # Get a download stream for the file
            download_stream = model_bucket.open_download_stream(model_id)
        except:
            continue

        # Read the contents of the stream into a bytes object
        model_bytes = download_stream.read()

        # Deserialize the model using pickle
        model = pickle.loads(model_bytes)

        start = doc.get("latest_timestamp") if doc.get("latest_timestamp") else db["1h"].find({'symbol': symbol}).sort('timestamp', pymongo.DESCENDING).limit(1)[0]['timestamp']
        params = {
            'interval': '1h',
            'limit':1000
        }
        params['symbol'] = symbol
        params['interval'] = interval

        # Set the start and end timestamps
        params['startTime'] = int(start)

        # Send the request and get the response
        response = requests.get("https://api.binance.com/api/v3/klines", params=params)

        # Check the status code of the response
        if response.status_code == 200:

            # Convert the response to a dictionary
            data = response.json()

            data = data[:-1]
            # Convert the data to a DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume', 'ignored'])

            # Set the timestamp column as the index
            # df.set_index('timestamp', drop=False, inplace=True)

            # Convert the timestamp column to a datetime
            # df.index = pd.to_datetime(df.index, unit='ms')
            buy_x,buy_y,sell_x,sell_y = [],[],[],[]
            new_close = []
            all_new_rows = []
            predicted_price = None
            old_predict = None
            old_actual_price =None
            symbol_portfolio = CryptoPortfolio()
            for timestamp,row in df.iterrows():
                all_new_rows.append(row.to_dict())
                new_close.append(float(row["close"]))
                # Subtract one day from the timestamp
                # close_timestamp =timestamp - timedelta(hours=1)
                input = [[float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),float(row["volume"])]]
                # if klines:
                #     for indicator in indicator_settings:
                #         func = eval(indicator.function)
                #         func([closes,new_close],**indicator.kwargs)
                if scaler_x:
                    input = getXWithWindowTransformed(klines +all_new_rows, scaler_x, window_size)
                    # input = scaler_x.transform([[float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]), 0.3]])
                if predicted_price:
                    old_predict = predicted_price
                try:
                    # Predict the close price for the next interval
                    predicted_price = model.predict(input)
                except:
                    continue

                predicted_price = scaler_y.inverse_transform([predicted_price])[0][-1]

                if old_predict and timestamp + 1 < len(df):
                    if predicted_price > old_predict:
                        symbol_portfolio.add_buy_order(symbol,(invest_in_usd_each_symbol/actual_price) * percent_change(old_predict,predicted_price),actual_price)
                        total_protfolio.add_buy_order(symbol, (invest_in_usd_each_symbol/actual_price) * percent_change(old_predict,predicted_price), actual_price)
                        buy_x.append(timestamp)
                        buy_y.append(predicted_price)
                    elif predicted_price < old_predict:
                        symbol_portfolio.sell_all(symbol,actual_price)
                        total_protfolio.sell_all(symbol, actual_price)
                        sell_x.append(timestamp)
                        sell_y.append(predicted_price)

                actual_price = df.shift(-1).iloc[timestamp]['close']
                if actual_price:
                    actual_price = float(actual_price)

                # if old_actual_price and actual_price:
                #     actual_prices.append(((actual_price - old_actual_price) / old_actual_price) * 100)

                # if predicted_price < row["close"]:
                #     plt.scatter(buy_x, buy_y, c='g', marker='^')
                # Get the actual price from the dataframe


                predicted_prices.append(predicted_price)
                if actual_price:
                    actual_price = float(actual_price)

                    # Add the predicted price to the list

                    # Add the actual price to the list
                    actual_prices.append(float(actual_price))

                    # input = scaler_x.transform([[float(row["open"]), float(row["high"]), float(row["low"]),
                    #                            float(row["close"]), float(row["volume"]), actual_price]])
                    if retrain_model:
                        all_input,y = getAllDataWithWindowTransformed(klines + all_new_rows,scaler_x,window_size)
                        actual_transform_price = scaler_y.transform([[actual_price]])
                        try:
                            model = type(model)()
                        except:
                            try:
                                model = type(model)(**model.get_params())
                            except:
                                try:
                                    model = model.clone()
                                except:
                                    old_model = model
                                    model = type(model)(estimators=old_model.estimators,weights=old_model.weights)
                                    # model.weights = old_model.weights
                                    # model.voting = old_model.voting
                        model.fit(all_input,y.ravel())

            symbol_portfolio.sell_all(symbol,actual_prices[-1])
            total_protfolio.sell_all(symbol,actual_prices[-1])
            symbol_portfolio.save_to_json(f"{trade_report_dir}/{symbol}_{interval}.json")

                    # Plot the predicted prices and actual prices
            # plt.scatter(buy_x, buy_y, c='g', marker='^', label='Buy')
            # plt.scatter(sell_x, sell_y, c='r', marker='v', label='Sell')
            #
            # plt.plot(predicted_prices, label='Predicted')
            # plt.plot(actual_prices, label='Actual')
            # # Add a legend and show the plot
            # plt.legend()
            # plt.show()
total_protfolio.save_to_json(f"{trade_report_dir}/total_portfolio.json")


