import streamlit as st
from sklearn.linear_model import LinearRegression

def linear_regression_param_selector():
    fit_intercept = st.checkbox("fit_intercept", value=True)
    copy_X = st.checkbox("copy_X", value=True)
    n_jobs = st.number_input("n_jobs", min_value=1, step=1, value=1)

    # Checkboxes don't have value 'False' when not checked, so we need to adjust the parameters accordingly
    normalize = st.checkbox("normalize", value=False)
    if normalize:
        params = {
            "fit_intercept": fit_intercept,
            "normalize": normalize,
            "copy_X": copy_X,
            "n_jobs": n_jobs
        }
    else:
        params = {
            "fit_intercept": fit_intercept,
            "copy_X": copy_X,
            "n_jobs": n_jobs
        }

    # Create and return the LinearRegression model with the specified parameters
    return LinearRegression(**params)
