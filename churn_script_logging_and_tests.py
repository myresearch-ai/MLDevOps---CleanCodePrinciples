"""This module contains unit tests for the churn_library.py functions.

Author: Ed Mwanza

Date: 5/16/22"""

import os
import logging
from math import ceil
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    data = cls.import_data("./data/bank_data.csv")
    try:
        data = cls.perform_eda(data)
        logging.info("SUCCESS: Testing perform_eda was successful")
    except KeyError as err:
        logging.error(
            "ERROR: Input data column '%s' was not found",
            err.args[0])
        raise err

    # Assert if churn_distribution.png is created
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png")
        logging.info(
            'SUCCESS: File %s was successfully found',
            'churn_distribution.png')
    except AssertionError as err:
        logging.error('ERROR: No such file was found on disk')
        raise err

    # Assert if customer_age_distribution.png is created
    try:
        assert os.path.isfile("./images/eda/customer_age_distribution.png")
        logging.info(
            'SUCCESS: File %s was successfully found',
            'customer_age_distribution.png')
    except AssertionError as err:
        logging.error('ERROR: No such file was found on disk')
        raise err

    # Assert if marital_status_distribution.png is created
    try:
        assert os.path.isfile("./images/eda/marital_status_distribution.png")
        logging.info(
            'SUCCESS: File %s was successfully found',
            'marital_status_distribution.png')
    except AssertionError as err:
        logging.error('ERROR: No such file was found on disk')
        raise err

    # Assert if total_transaction_distribution.png is created
    try:
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png")
        logging.info(
            'SUCCESS: File %s was successfully found',
            'total_transaction_distribution.png')
    except AssertionError as err:
        logging.error('ERROR: No such file was found on disk')
        raise err

    # Assert if corr.png is created
    try:
        assert os.path.isfile("./images/eda/corr.png")
        logging.info('SUCCESS: File %s was found', 'corr.png')
    except AssertionError as err:
        logging.error('ERROR: No such file was found on disk')
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    # Load required DataFrame
    data = cls.import_data("./data/bank_data.csv")

    # Create Churn feature
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Categorical Features
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']

    # Perform cheks
    try:
        enc_df = cls.encoder_helper(
            data=data,
            category_lst=[],
            response=None)

        # Data is expcted to be the same
        assert enc_df.equals(data)
        logging.info(
            "SUCCESS: Testing encoder_helper(data_frame, category_lst=[]) succeeded.")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing encoder_helper(data_frame, category_lst=[]) failed.")
        raise err

    try:
        enc_df = cls.encoder_helper(
            data=data,
            category_lst=category_lst,
            response=None)

        # Column names expected to be same
        assert enc_df.columns.equals(data.columns)

        # Data expected to be different
        assert enc_df.equals(data) is False
        logging.info(
            "SUCCESS: Testing encoder_helper(data_frame, \
                category_lst=category_lst, response=None) succeeded.")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing encoder_helper(data_frame, \
                category_lst=category_lst, response=None) failed.")
        raise err

    try:
        enc_df = cls.encoder_helper(
            data=data,
            category_lst=category_lst,
            response='Churn')

        # Columns names expected to be different
        assert enc_df.columns.equals(data.columns) is False

        # Data expected to be different
        assert enc_df.equals(data) is False

        # Length of cols in enc_df is expcted to be the sum of length of cols
        # in original & category_lst
        assert len(enc_df.columns) == len(data.columns) + len(category_lst)
        logging.info(
            "SUCCESS: Testing encoder_helper(data_frame, \
                category_lst=category_lst, response='Churn') succeeded.")
    except AssertionError as err:
        logging.error(
            "ERROR: Testing encoder_helper(data_frame, \
                category_lst=category_lst, response='Churn') failed.")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    # Load the required DataFrame
    data = cls.import_data("./data/bank_data.csv")

    # Create Churn feature
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Perform checks
    try:
        (_, x_test, _, _) = cls.perform_feature_engineering(
            data=data,
            response='Churn')

        # Churn feature is expected to be in data
        assert 'Churn' in data.columns
        logging.info(
            "SUCCESS: Testing perform_feature_engineering - \
                Churn feature was successfully found in the data.")
    except KeyError as err:
        logging.error(
            'ERROR: The Churn feature was not successfully found in the data.')
        raise err

    try:
        # x_test size must be 30% of the data after splitting
        assert x_test.shape[0] == ceil(data.shape[0] * 0.3)
        logging.info(
            'SUCCESS: Testing perform_feature_engineering - \
                x_test & data sizes are consistent.')
    except AssertionError as err:
        logging.error(
            'ERROR: Testing perform_feature_engineering - \
                x_test & data sizes are inconsistent.')
        raise err


def test_train_models():
    '''
    test train_models
    '''
    # Load the required data
    data = cls.import_data("./data/bank_data.csv")

    # Create Churn feature
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Perform feature engineering
    (x_train, x_test, y_train, y_test) = cls.perform_feature_engineering(
        data=data,
        response='Churn')

    # Assert if logistic_model.pkl file exists in designated directory
    try:
        cls.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl")
        logging.info(
            'SUCCESS: The pickled model file %s was successfully found.',
            'logistic_model.pkl')
    except AssertionError as err:
        logging.error(
            'ERROR: The pickled model file was not found in designated directory.')
        raise err

    # Assert if rfc_model.pkl file exists in designated directory
    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info(
            'SUCCESS: The pickled model file %s was successfully found.',
            'rfc_model.pkl')
    except AssertionError as err:
        logging.error(
            'ERROR: The pickled model file was not found in designated directory.')
        raise err

    # Assert if roc_curve_result.png file exists in designated directory
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png')
        logging.info(
            'SUCCESS: The image file %s was successfully found.',
            'roc_curve_result.png')
    except AssertionError as err:
        logging.error(
            'ERROR: The image file was not found in designated directory.')
        raise err

    # Assert if RF images exists in designated directory
    try:
        assert os.path.isfile('./images/results/RandomForestTrain.png')
        logging.info(
            'SUCCESS: The image file %s was successfully found.',
            'RandomForestTrain.png')
    except AssertionError as err:
        logging.error(
            'ERROR: The image file was not found in designated directory.')
        raise err


    # Assert if LR images exists in designated directory
    try:
        assert os.path.isfile('./images/results/LogRegTrain.png')
        logging.info(
            'SUCCESS: The image file %s was successfully found.',
            'LogRegTrain.png')
    except AssertionError as err:
        logging.error(
            'ERROR: The image file was not found in designated directory.')
        raise err


    # Assert if feature_importances.png file exists in designated directory
    try:
        assert os.path.isfile('./images/results/feature_importances.png')
        logging.info(
            'SUCCESS: The pickled model file %s was successfully found.',
            'feature_importances.png')
    except AssertionError as err:
        logging.error(
            'ERROR: The pickled model file was not found in designated directory.')
        raise err


if __name__ == '__main__':
    # Run the tests
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
