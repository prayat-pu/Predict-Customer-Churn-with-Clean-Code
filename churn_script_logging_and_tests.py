"""
Objectives: This file is used to test all function of churn_library.py
Edit Date: 06/01/2024
Author: Prahyat
"""

import os
import logging
import churn_library as churn_lib

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing import_eda: The file wasn't found")

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    try:
        dataframe = churn_lib.import_data("./data/bank_data.csv")
        perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except BaseException:
        logging.error("Testing perform_eda: FAILED")

    try:
        churn_distribution_filepath = os.path.join(
            './images/eda/', 'Churn_distribution.png')
        assert os.path.isfile(churn_distribution_filepath)
    except AssertionError:
        logging.error(
            "Testing perform_eda: The churn distribution file wasn't found")

    try:
        customer_age_distribution_filepath = os.path.join(
            './images/eda/', 'Customer_Age_distribution.png')
        assert os.path.isfile(customer_age_distribution_filepath)
    except AssertionError:
        logging.error(
            "Testing perform_eda: The customer age distribution file wasn't found")

    try:
        matital_status_filepath = os.path.join(
            './images/eda/', 'count_by_matital_status.png')
        assert os.path.isfile(matital_status_filepath)
    except AssertionError:
        logging.error(
            "Testing perform_eda: The matital status file wasn't founds")

    try:
        total_transCT_filepath = os.path.join(
            './images/eda/', 'total_trans_Ct_distribution.png')
        assert os.path.isfile(total_transCT_filepath)
    except AssertionError:
        logging.error(
            "Testing perform_eda: The total trans CT distribution file wasn't found")

    try:
        feature_correlation_filepath = os.path.join(
            './images/eda/', 'features_correlation.png')
        assert os.path.isfile(feature_correlation_filepath)
    except AssertionError:
        logging.error(
            "Testing perform_eda: feature correlation file wasn't found")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        dataframe = churn_lib.import_data("./data/bank_data.csv")
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        response = 'Churn'

        dataframe = encoder_helper(dataframe, category_lst, response)

        logging.info("Testing encoder_helper: SUCCESS")
    except BaseException:
        logging.error("Testing encoder_helper: FAILED: Something Wrong")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    try:
        dataframe = churn_lib.import_data("./data/bank_data.csv")
        response = 'Churn'

        perform_feature_engineering(dataframe, response)

        logging.info("Testing perform_feature_engineering: SUCCESS")
    except KeyError:
        logging.error(
            "Testing perform_feature_engineering: FAILED with KeyError Some columns not in index")


def test_train_models(train_models):
    '''
    test train_models
    '''

    try:
        dataframe = churn_lib.import_data("./data/bank_data.csv")
        response = 'Churn'

        x_training, x_testing, y_training, y_testing = churn_lib.perform_feature_engineering(
            dataframe, response)

        train_models(x_training, x_testing, y_training, y_testing)

        logging.info("Testing train_models: SUCCESS")
    except BaseException:
        logging.error("Testing train_models: FAILED: Something Wrong")

    try:
        roc_curve_filepath = os.path.join(
            './images/results/', 'rfc_lrc_roc_curve_plot.png')
        assert os.path.isfile(roc_curve_filepath)
    except AssertionError:
        logging.error("Testing train_models: roc curve file wasn't found")

    try:
        feature_importance_filepath = os.path.join(
            './images/results/', 'rfc_feature_importance.png')
        assert os.path.isfile(feature_importance_filepath)
    except AssertionError:
        logging.error(
            "Testing train_models: feature importance file wasn't found")

    try:
        rfc_model_filepath = os.path.join('./models/', 'rfc_model.pkl')
        assert os.path.isfile(rfc_model_filepath)
    except AssertionError:
        logging.error("Testing train_models: rfc model file wasn't found")

    try:
        log_model_filepath = os.path.join('./models/', 'logistic_model.pkl')
        assert os.path.isfile(log_model_filepath)
    except AssertionError:
        logging.error("Testing train_models: logistic model file wasn't found")

    try:
        rfc_cls_filepath = os.path.join(
            './images/results/',
            'random_forest_cls_matrix.png')
        assert os.path.isfile(rfc_cls_filepath)
    except AssertionError:
        logging.error(
            "Testing train_models: random forest classification metrix file wasn't found")

    try:
        log_cls_filepath = os.path.join(
            './images/results/',
            'logistic_regression_cls_matrix.png')
        assert os.path.isfile(log_cls_filepath)
    except AssertionError:
        logging.error(
            "Testing train_models: logistic classification metrix file wasn't found")


if __name__ == "__main__":
    test_import(churn_lib.import_data)
    test_eda(churn_lib.perform_eda)
    test_encoder_helper(churn_lib.encoder_helper)
    test_perform_feature_engineering(churn_lib.perform_feature_engineering)
    test_train_models(churn_lib.train_models)
