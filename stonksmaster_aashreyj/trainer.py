import datetime as dt

import joblib
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from utils import prepare_data

# define constants to be used throughout the app
TOKEN = 'pk_1ff26bf4c61946f2a1df7f8efc776b9d'
API_RANGE = '1d'
LIBRARY_PERIOD = '2y'
DEFAULT_START_TIME = '09:30'
SEED = 5


# generate data from yfinance library
def generate_historical_data(ticker):
    # generate dataframe from yfinance library
    finance_df = yf.download(ticker, period=LIBRARY_PERIOD)

    # remove the useless columns from the dataframe
    finance_df = finance_df.loc[:, ['Open']]

    # convert date field of each row to corresponding timestamp
    # and create new column containing these timestamp values
    timestamps = list()
    finance_df.index.to_pydatetime()

    for index, row in finance_df.iterrows():
        time = dt.datetime.strptime(DEFAULT_START_TIME, '%H:%M')
        timestamp = dt.datetime.timestamp(dt.datetime.combine(index.date(), time.time()))
        timestamps.append(timestamp)
    finance_df.insert(0, 'Timestamp', timestamps)

    # call reset_index() method to get integral indices
    finance_df.reset_index(inplace=True, drop=True)

    return finance_df


# generate data from historical data API
def generate_historical_api_data(url):
    # generate dataframe from historical data API
    prev_day_data = requests.get(url).json()
    prev_df = pd.DataFrame(prev_day_data)

    # remove useless columns from the dataframe
    prev_df = prev_df.loc[:, ['date', 'minute', 'open']]

    # combine date and time fields of each row to generate
    # corresponding timestamp and create new column containing
    # these timestamp values
    historic_timestamps = list()
    for index, row in prev_df.iterrows():
        date = dt.datetime.strptime(row['date'], '%Y-%m-%d')
        time = dt.datetime.strptime(row['minute'], '%H:%M')
        timestamp = dt.datetime.timestamp(dt.datetime.combine(date.date(), time.time()))
        historic_timestamps.append(timestamp)

    # we require only the opening prices of the stocks along with
    # the timestamps, so other columns can be dropped
    prev_df = prev_df.loc[:, ['open']]
    prev_df.rename(columns={'open': 'Open'}, inplace=True)
    prev_df.insert(0, 'Timestamp', historic_timestamps)

    return prev_df


# generate intra-day data from API
def generate_intra_day_data(url):
    # generate dataframe from intra-day data API
    intra_day_data = requests.get(url).json()
    curr_df = pd.DataFrame(intra_day_data)

    # remove useless columns from the dataframe
    curr_df = curr_df.loc[:, ['date', 'minute', 'open']]

    # combine date and time fields of each row to generate
    # corresponding timestamp and create new column containing
    # these timestamp values
    current_timestamps = list()
    for index, row in curr_df.iterrows():
        date = dt.datetime.strptime(row['date'], '%Y-%m-%d')
        time = dt.datetime.strptime(row['minute'], '%H:%M')
        timestamp = dt.datetime.timestamp(dt.datetime.combine(date.date(), time.time()))
        current_timestamps.append(timestamp)

    # we require only the opening prices of the stocks along with
    # the timestamps, so other columns can be dropped
    curr_df = curr_df.loc[:, ['open']]
    curr_df.rename(columns={'open': 'Open'}, inplace=True)
    curr_df.insert(0, 'Timestamp', current_timestamps)

    return curr_df


# this function is used only while training
# so that trained models can be dumped to files
# in order to make future predictions faster
def app_training(ticker_symbol):
    url_intra_day = f'https://cloud.iexapis.com/stable/stock/{ticker_symbol}/intraday-prices/batch?token={TOKEN}'
    url_historical = f'https://cloud.iexapis.com/stable/stock/{ticker_symbol}/chart/{API_RANGE}?token={TOKEN}'

    finance_df = generate_historical_data(ticker_symbol)
    prev_df = generate_historical_api_data(url=url_historical)
    curr_df = generate_intra_day_data(url=url_intra_day)

    finance_df.append(prev_df, ignore_index=True)
    finance_df.append(curr_df, ignore_index=True)

    x_train, x_test, y_train, y_test = prepare_data(finance_df)

    # train the model with available data and return the trained model
    trained_model = GradientBoostingRegressor(n_estimators=100)
    trained_model.fit(x_train, y_train)

    # dump financial data to csv file
    finance_df.to_csv(f'{ticker_symbol.lower()}_data.csv')

    return trained_model


# if this module is executed, the models are trained and dumped to files
if __name__=='__main__':
    model1 = app_training('MSFT')
    model2 = app_training('AAPL')
    model3 = app_training('NFLX')
    model4 = app_training('TTM')
    model5 = app_training('NVDA')
    model6 = app_training('INTC')

    joblib.dump(model1, 'msft_model.joblib')
    joblib.dump(model2, 'aapl_model.joblib')
    joblib.dump(model3, 'nflx_model.joblib')
    joblib.dump(model4, 'ttm_model.joblib')
    joblib.dump(model5, 'nvda_model.joblib')
    joblib.dump(model6, 'intc_model.joblib')