import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from ml_pipeline import model_performance
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.feature_selection import RFE

# Function to split data into train and test sets
def split_data(data, target, size, randomstate):
    # Split data into X (features) and y (target)
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=size, 
        random_state=randomstate
    )
    return X_train, X_test, y_train, y_test

# Function to process data for Linear Regression
def process_data_for_LR(X_train, X_test, y_train, y_test):
    # Drop unnecessary columns
    cols_to_drop = ['children', 'region', 'sex']
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)
    
    # Encode categorical variables
    ohe = OneHotEncoder(use_cat_names=True)
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)
    
    # Drop redundant feature
    cols_to_drop = ['smoker_no']
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    # Transform target variable
    pt = PowerTransformer(method='yeo-johnson')
    y_train_t = pt.fit_transform(y_train.values.reshape(-1, 1))[:, 0]
    y_test_t = pt.transform(y_test.values.reshape(-1, 1))[:, 0]

    return X_train, X_test, y_train, y_test, y_train_t, y_test_t, pt

# Function to train and evaluate Linear Regression
def train_and_evaluate_LR(X_train, X_test, y_train, y_test, y_train_t, y_test_t, pt):
    # Compute sample weights
    sample_weight = y_train / y_train.min()

    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(
        X_train, 
        y_train_t, 
        sample_weight=sample_weight
    )

    # Evaluate model
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    # Inverse transform target predictions
    y_pred_train = pt.inverse_transform(y_pred_train.reshape(-1, 1))[:, 0]
    y_pred_test = pt.inverse_transform(y_pred_test.reshape(-1, 1))[:, 0]

    # Calculate and print model performance
    base_perf_train = model_performance.calc_model_performance(y_train, y_pred_train)
    base_perf_test = model_performance.calc_model_performance(y_test, y_pred_test)

    print('Linear Regression Results for Training set')
    print(base_perf_train)
    print(" ")
    print('Linear Regression Results for Testing set')
    print(base_perf_test)

    return y_pred_train, y_pred_test, lr

# Function to process data for XGBoost modeling
def process_data_for_xgboost(X_train, X_test):
    # Encode categorical variables
    ohe = OneHotEncoder(use_cat_names=True)
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)

    return X_train, X_test, ohe

# Function to train and evaluate XGBoost using Bayesian search
def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test):
    rfe = RFE(estimator=XGBRegressor())
    xgb = XGBRegressor()

    steps = [
        ('rfe', rfe),
        ('xgb', xgb)
    ]

    pipe = Pipeline(steps)

    num_features = X_train.shape[1]
    search_spaces = {
        'rfe__n_features_to_select': Integer(1, num_features),
        'xgb__n_estimators': Integer(1, 500),
        'xgb__max_depth': Integer(2, 8),
        'xgb__reg_lambda': Integer(1, 200),
        'xgb__learning_rate': Real(0, 1),
        'xgb__gamma': Real(0, 2000)
    }

    xgb_bs_cv = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_spaces,
        scoring='neg_root_mean_squared_error',
        n_iter=75,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=0
    )

    xgb_bs_cv.fit(X_train, y_train)

    y_pred_train_xgb = xgb_bs_cv.predict(X_train)
    y_pred_test_xgb = xgb_bs_cv.predict(X_test)

    xgb_perf_train = model_performance.calc_model_performance(y_train, y_pred_train_xgb)
    xgb_perf_test = model_performance.calc_model_performance(y_test, y_pred_test_xgb)

    print('XGBoost Results for Training set')
    print(xgb_perf_train)
    print(" ")
    print('XGBoost Results for Testing set')
    print(xgb_perf_test)

    return y_pred_train_xgb, y_pred_test_xgb, xgb_bs_cv
