# library doc string
"""
Objective: This file is used to train a model for churn predicting.
Edit Date: 06/01/2024
Author: Prahyat
"""
# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import seaborn as sns
# from sklearn.preprocessing import normalize

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data_df = pd.read_csv(pth)
    data_df['Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_df


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    image_pth = './images/eda/'

    # perform distribution checking for

    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(image_pth + 'Churn_distribution.png')
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(image_pth + 'Customer_Age_distribution.png')
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(image_pth + 'count_by_matital_status.png')
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(image_pth + 'total_trans_Ct_distribution.png')
    # correlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(image_pth + 'features_correlation.png')


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]
    output:
            dataframe: pandas dataframe with new columns for
    '''

    for col in category_lst:
        col_list = []
        col_groups = dataframe.groupby(col).mean()[response]

        for val in dataframe[col]:
            col_list.append(col_groups.loc[val])

        dataframe[col + '_Churn'] = col_list

    return dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]
    output:
              feature_train: X training data
              feature_test: X testing data
              label_train: y training data
              label_test: y testing data
    '''

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    dataframe = encoder_helper(dataframe, category_lst, response)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    empty_df = pd.DataFrame()
    empty_df[keep_cols] = dataframe[keep_cols]
    label_y = dataframe['Churn']
    feature_train, feature_test, label_train, label_test = train_test_split(
        empty_df, label_y, test_size=0.3, random_state=42)
    return feature_train, feature_test, label_train, label_test


def classification_report_image(label,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            label: training and test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
#     print('random forest results')
    label_train, label_test = label[0],label[1]
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(label_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(label_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.savefig('./images/results/random_forest_cls_matrix.png')

#     print('logistic regression results')
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(label_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(label_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.savefig('./images/results/logistic_regression_cls_matrix.png')


def feature_importance_plot(model, feature_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            feature_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

#     # features important with SHAP
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(feature_data)
#     shap.summary_plot(shap_values, feature_data, plot_type="bar")

#     plt.savefig(output_pth+'_SHAP.png')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [feature_data.columns[i] for i in indices]
    # Add bars
    plt.bar(range(feature_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(feature_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth + '.png')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5)  # create grid search for random forest
    cv_rfc.fit(x_train, y_train)  # train random forest with grid search

    lrc.fit(x_train, y_train)  # train logistic regression



    y_train_preds_rf = cv_rfc.best_estimator_.predict(
        x_train)  # rf predict x_train with best parameters
    y_test_preds_rf = cv_rfc.best_estimator_.predict(
        x_test)  # rf predict x_test with best parameters

    # logistics predict x_train with best parameters
    y_train_preds_lr = lrc.predict(x_train)
    # logistics predict x_test with best parameters
    y_test_preds_lr = lrc.predict(x_test)

    # report cls image into local path
    classification_report_image([y_train,y_test],
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


    # plots
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)

    plt.figure(figsize=(15, 8))
    roc_plot_ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=roc_plot_ax,
        alpha=0.8)
    lrc_plot.plot(ax=roc_plot_ax, alpha=0.8)
    plt.savefig('./images/results/rfc_lrc_roc_curve_plot.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
#     lr_model = joblib.load('./models/logistic_model.pkl')



    # plot features important with SHAP values
    feature_importance_plot(
        rfc_model,
        x_test,
        './images/results/rfc_feature_importance')



if __name__ == "__main__":
    DATA_PTH = './data/bank_data.csv'
    dataframe_df = import_data(DATA_PTH)
    perform_eda(dataframe_df)
    X_training, X_testing, y_training, y_testing = perform_feature_engineering(
        dataframe_df, 'Churn')
    train_models(X_training, X_testing, y_training, y_testing)

