import numpy as np
from sklearn.model_selection import train_test_split


# the purpose of this function is to scale the input data
# and split it into training and testing datasets
def prepare_data(finance_df):
    # remove infinite and nan values from the dataset
    finance_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    finance_df.dropna(inplace=True)

    # split input and output columns
    X = finance_df.drop(columns=['Open'])
    y = finance_df.loc[:, 'Open']

    # split training and testing data sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    return x_train, x_test, y_train, y_test