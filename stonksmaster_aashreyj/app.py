import datetime as dt

import joblib
import pandas as pd
from utils import prepare_data


# this function makes the predictions and returns the dataframe to be plotted
def make_predictions(model, x_test, y_test):
    # make predictions using the model
    predictions = model.predict(x_test)

    # create a datetime index using the timestamps
    # this  gives higher readability
    datetime_index = []
    for timestamp in x_test.values:
        datetime_index.append(dt.datetime.fromtimestamp(int(timestamp)))

    # create new dataframe containing the values to be plotted
    # and make timestamps as its index
    predictions_df = pd.DataFrame(datetime_index, columns=['Timestamp'])
    predictions_df.insert(1, 'Actual Price', y_test.values)
    predictions_df.insert(2, 'Predicted Price', predictions)
    predictions_df.set_index('Timestamp', inplace=True, drop=True)

    # sort according to datetime values
    predictions_df.sort_index(inplace=True)

    return predictions_df


# this is the main function of the app that returns the predictions
# made by the trained model
def app(ticker_symbol):

    # tries to read the corresponding CSV file
    try:
        finance_df = pd.read_csv(f'{ticker_symbol.lower()}_data.csv')
    except FileNotFoundError:
        return -1

    # read appropriate columns and drop irrelevant ones
    finance_df.columns = ['Index', 'Timestamp', 'Open']
    finance_df.drop(columns=['Index'], inplace=True)

    # prepare data
    x_train, x_test, y_train, y_test = prepare_data(finance_df)

    # choose model to make predictions
    try:
        model = joblib.load(f'{ticker_symbol.lower()}_model.joblib')
    except FileNotFoundError:
        return -1

    # return dataframe that contains the predictions made by the model
    return make_predictions(model=model, x_test=x_test, y_test=y_test)
