import numpy as np
import pandas as pd  # import pandas
# import os
from sklearn import preprocessing
# from scipy.sparse import csr_matrix
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def preprocess_data():
    train_df_insurance = pd.read_csv('/home/reeta/Desktop/Machine_learning_1260/home_insurance_quote_flag/train.csv')
    train_df_insurance = train_df_insurance.drop('QuoteNumber', axis=1)

# Find correlation between features and drop the one of two highly correlated one
    def highly_corr_col(abc):
        corr_matrix = train_df_insurance[abc].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        return to_drop

    field_col = [col for col in train_df_insurance if col.startswith('Field')]
    coveragefield_col = [col for col in train_df_insurance if col.startswith('CoverageField')]
    salesfield_col = [col for col in train_df_insurance if col.startswith('SalesField')]
    personalfield_col = [col for col in train_df_insurance if col.startswith('PersonalField')]
    propertyfield_col = [col for col in train_df_insurance if col.startswith('PropertyField')]
    geographicfield_col = [col for col in train_df_insurance if col.startswith('GeographicField')]
    train_df_insurance.drop(highly_corr_col(field_col), axis=1, inplace=True)
    train_df_insurance.drop(highly_corr_col(coveragefield_col), axis=1, inplace=True)
    train_df_insurance.drop(highly_corr_col(salesfield_col), axis=1, inplace=True)
    train_df_insurance.drop(highly_corr_col(personalfield_col), axis=1, inplace=True)
    train_df_insurance.drop(highly_corr_col(propertyfield_col), axis=1, inplace=True)
    train_df_insurance.drop(highly_corr_col(geographicfield_col), axis=1, inplace=True)

    # Convert str_type 'Date' into date_type
    train_df_insurance['Date'] = pd.to_datetime(pd.Series(train_df_insurance['Original_Quote_Date']))

    # Drop 'Original_Quote_date'
    train_df_insurance = train_df_insurance.drop('Original_Quote_Date', axis=1)

    # Extract year,month,weekday from 'Date'
    train_df_insurance['Year'] = train_df_insurance['Date'].apply(lambda a: a.year)
    train_df_insurance['Month'] = train_df_insurance['Date'].apply(lambda a: a.month)
    train_df_insurance['weekday'] = train_df_insurance['Date'].apply(lambda a: a.weekday())
    # Drop 'Date' feature
    train_df_insurance = train_df_insurance.drop('Date', axis=1)

    # Fill Missing Nan Value For Categorical Columns With 'unknown' and other -1:
    # CHECK MISSING DATA
    # Let us organize above table and sort the table in terms of # of NAN in descending order
    nan_info = pd.DataFrame(train_df_insurance.isnull().sum()).reset_index()
    nan_info.columns = ['feature_name', 'nan_cnt']
    nan_info.sort_values(by='nan_cnt', ascending=False, inplace=True)
    nan_info['nan_percentage'] = nan_info['nan_cnt'] / len(train_df_insurance)
    # print(nan_info.head(10))
    # get all cols with missing data
    cols_with_missing = nan_info.loc[nan_info.nan_cnt > 0].feature_name.values
    # print(cols_with_missing)
    features = [f for f in train_df_insurance.columns.values if
                f not in ['QuoteConversion_Flag']]  # you have to customize this according to your own needs

    # print(features)
    def enc(c):
        le = preprocessing.LabelEncoder()
        le.fit(list(c.values))
        c = le.transform(list(c.values))
        return c

    for ft in cols_with_missing:
        if train_df_insurance[ft].dtype == 'object':
            train_df_insurance[ft].fillna('unknown', inplace=True)
        else:
            train_df_insurance[ft].fillna(-1, inplace=True)
        # print(df[ft])

        # Using Label encoder
        enc(train_df_insurance[ft])
        # Convert all strings to equivalent numeric representations:
        for f in train_df_insurance.columns:
            if train_df_insurance[f].dtype == 'object':
                enc(train_df_insurance[f])

        # To Find Hidden Categorical features
    category_features = []
    f_cat = []
    threshold = 70
    for each in features:

        if train_df_insurance[each].nunique() < threshold:
            category_features.append(each)
    for each in category_features:
        train_df_insurance[each] = train_df_insurance[each].astype('category')
        # df_cat.append(each)
        # print(df[each])
        enc(train_df_insurance[each])
        f_cat.append(each)
    # category_features
    x = csr_matrix(pd.get_dummies(train_df_insurance[f_cat], drop_first=True, prefix=f_cat, sparse=True)).tocsr()


# split the data by train test split
    x = x.toarray()
    y = train_df_insurance['QuoteConversion_Flag'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=10, test_size=0.1)
    print(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test


preprocess_data()
