"""This modules demonstrates implementation of a simple end-to-end
machine learning implementation using Python PEP8 best coding and
practices

Author: Ed Mwanza

Date: 5/16/22"""


# import libraries
import os
from typing import Optional, List, Tuple
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''
    try:
        data = pd.read_csv(pth)
        return data
    except FileNotFoundError as err:
        raise err


def perform_eda(data: pd.DataFrame, figsize=(20, 10)) -> pd.DataFrame:
    '''
    perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            data_out: pandas dataframe
    '''
    data_eda = data.copy(deep=True)

    # Churn
    plt.figure(figsize=figsize)
    data_eda['Churn'] = data_eda['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Churn Distribution
    plt.figure(figsize=figsize)
    data_eda['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # Customer Age Distribution
    plt.figure(figsize=figsize)
    data_eda['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Normalized Marital Status Distribution
    plt.figure(figsize=figsize)
    data_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Total Transaction Distribution
    plt.figure(figsize=figsize)
    sns.histplot(data_eda, x='Total_Trans_Ct', stat='density', kde=True)
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    # Correlation map
    plt.figure(figsize=figsize)
    sns.heatmap(data_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/corr.png')

    return data_eda


def encoder_helper(
        data: pd.DataFrame,
        category_lst: List[str],
        response: Optional[str] = None) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        data: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could
        be used for naming variables or index y column]
    output:
        data: pandas dataframe with new columns for
    '''
    data_enc = data.copy(deep=True)
    for category in category_lst:
        column_lst = []
        column_groups = data.groupby(category).mean()['Churn']

        for val in data[category]:
            column_lst.append(column_groups.loc[val])
        if response is not None:
            data_enc[category + '_' + response] = column_lst
        else:
            data_enc[category] = column_lst

    return data_enc


#data_returns = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
def perform_feature_engineering(data: pd.DataFrame,
                                response: Optional[str] = None) -> Tuple[pd.DataFrame,
                                                                         pd.DataFrame,
                                                                         pd.DataFrame,
                                                                         pd.DataFrame]:
    '''
    input:
        data: pandas dataframe
        response: string of response name [optional argument that could be used
        for naming variables or index y column]
    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    # List of categorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    # Perform feature engineering
    enc_data = encoder_helper(
        data=data,
        category_lst=cat_columns,
        response=response)

    # Response feature
    response_var = enc_data['Churn']

    # Placeholder dataframe
    features = pd.DataFrame()

    # Columns to keep in features
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

    # Populate features with features
    features[keep_cols] = enc_data[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, response_var, test_size=0.3, random_state=42)

    return (
        x_train, x_test, y_train, y_test)

# Pylint penalty: too-many-args
# Correction: put the dataframes in - List[pd.DataFrame] then index into list.


def classification_report_image(y_train_test: List[pd.DataFrame],
                                y_train_preds_lr: List[float],
                                y_train_preds_rf: List[float],
                                y_test_preds_lr: List[float],
                                y_test_preds_rf: List[float]):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
    output:
        None
    '''
    # Extract y_train & y_test from y_train_test
    y_train = y_train_test[0]
    y_test = y_train_test[1]
    # RandomForestClassifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')

    # LogisticRegression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values
        output_pth: path to store the figure
    output:
        None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')


def train_models(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)

    # Parameters for Grid Search
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']}

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    # LogisticRegression
    lrc.fit(x_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    lrc_plot = plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_,
                              x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    rfc_disp.plot(ax=axis, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')

    # Create y_train_test list
    y_train_test = [y_train, y_test]

    # Compute and results
    classification_report_image(y_train_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            x_data=x_test,
                            output_pth='./images/results/')


if __name__ == '__main__':
    # Import data
    BANK_DF = import_data(pth='./data/bank_data.csv')

    # Perform EDA
    EDA_DF = perform_eda(data=BANK_DF)

    # Feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        data=EDA_DF, response='Churn')

    # Model training,prediction and evaluation
    train_models(x_train=X_TRAIN,
                 x_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST)
