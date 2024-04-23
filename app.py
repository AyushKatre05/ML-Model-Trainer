import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ui import model_selector, sidebar_controllers
from functions import *


def build_model(df, model):
    st.write("Data Set After preprossing ")
    st.write(df.head())
    X = df.iloc[:, :-1]  # Using all columns except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider(
            'Data split ratio (% for Training Set)', 10, 90, 80, 5)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    st.markdown('** Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('** Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    model.fit(X_train, Y_train)
    st.write('Model Info')
    st.info(model)

    st.markdown('** Training set**')
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    train_mse = mean_squared_error(Y_train, y_train_pred)
    test_mse = mean_squared_error(Y_test, y_test_pred)

    st.write('Train Mean Squared Error:', train_mse)
    st.write('Test Mean Squared Error:', test_mse)

    st.write('Model Parameters')
    st.write(model.get_params())


def scaling(df):
    # Assuming `y` is the target variable you want to concatenate with the scaled DataFrame `df_norm`
    # You should modify this line according to your specific use case
    y = pd.DataFrame()

    # Scale the DataFrame
    # Replace this line with your actual scaling logic
    df_norm = df

    # Concatenate df_norm and y along columns
    df_norm = pd.concat([df_norm, y], axis=1)  # Use square brackets to create a list of DataFrames

    return df_norm


def encodeing_df(df):
    col_name = []
    label_encoder = LabelEncoder()

    for colname, colval in df.items():  # Using items() method to iterate over columns
        if colval.dtype == 'object':
            col_name.append(colname)

    for col in col_name:
        df[col] = label_encoder.fit_transform(df[col])

    return df


if __name__ == "__main__":
    data_set, model_type, model, degree = sidebar_controllers()

    def replace_null(df):
        for colname, colval in df.items():
            if colval.isnull().sum() > 0:
                if colval.dtype in [np.float64, np.int64]:  # Check if the column contains numeric values
                    col_mean = colval.mean()
                    df[colname].fillna(col_mean, inplace=True)
        return df

    if data_set is not None:
        df = pd.read_csv(data_set)
        st.subheader(' Glimpse of dataset')
        st.write(df.head())

        # Check if df is a DataFrame
        if isinstance(df, pd.DataFrame):
            df = replace_null(df)
            df = encodeing_df(df)
            df = scaling(df)
            build_model(df, model)
        else:
            st.error("The uploaded file is not in the expected format. Please upload a CSV file.")
    else:
        st.info('Awaiting for CSV file to be uploaded.')
