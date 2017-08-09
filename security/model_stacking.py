# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest


RANDOM_SEED = 420


def _missing(df):
    print("数据总大小为: {}".format(df.shape))
    df_shape = df.shape
    df_samples = df_shape[0]
    df_columns = df_shape[1]
    missing_df = df.isnull()
    missing_columns = missing_df.sum(axis=0)
    missing_columns = missing_columns[missing_columns > 0].sort_values(ascending=False)
    if len(missing_columns) == 0:
        print("没有任何缺失数据")
        return
    else:
        missing_col_count = len(missing_columns)
        print("有{0}列数据缺失，占总列数的{1}%".format(missing_col_count, round(missing_col_count/df_columns, 6) * 100))
        print("缺失的列为: {}".format(missing_columns.index.values))
    missing_samples = missing_df.sum(axis=1)
    missing_samples = missing_samples[missing_samples > 0]
    missing_spl_count = len(missing_samples)
    print("有{0}行数据缺失，占总行数的{1}%".format(missing_spl_count, round(missing_spl_count/df_samples, 6) * 100))
    return


def _pearsonr_score(X, y):
    score_list = []
    p_list = []
    for col in X.T:
        score, p = pearsonr(col, y)
        score_list.append(score)
        p_list.append(p)
    return np.array(score_list)


def get_data(train_feature="train_x_2.csv", test_feature="test_x_2.csv", train_label="./data/df_id_train.csv", test_id="./data/df_id_test.csv"):
    train_feature = pd.read_csv(train_feature, encoding="gbk")
    test_feature = pd.read_csv(test_feature, encoding="gbk")
    train_label = pd.read_csv(train_label, header=None)
    test_label = pd.read_csv(test_id, header=None)

    column_name = ["f" + str(i) if i != 0 else "uid" for i in range(train_feature.shape[1])]
    train_feature.columns = column_name
    test_feature.columns = column_name
    train_label.columns = ["uid", "label"]
    test_label.columns = ["uid"]

    train = train_label.merge(train_feature, on="uid", how="left")
    test = test_label.merge(test_feature, on="uid", how="left")
    train_x = train.iloc[:, 2:]
    train_y = train.iloc[:, 1]
    test_x = test.iloc[:, 1:]
    test_uid = test.iloc[:, 0]

    return train_x, train_y, test_x, test_uid


def scale_data(train, test=None):
    std_scaler = StandardScaler()
    std_train = std_scaler.fit_transform(train)
    if test is not None:
        std_test = std_scaler.transform(test)
        return std_train, std_test, std_scaler
    return std_train, None, std_scaler


def grid_search(model, params, X, y, score_method="f1", cv=5, n_job=-1, verbose=0):
    g_model = GridSearchCV(model, param_grid=params, scoring=score_method, cv=cv, n_jobs=n_job, iid=False, verbose=verbose)
    g_model.fit(X, y)
    for t in g_model.grid_scores_:
        _params = t[0]
        _mean = t[1]
        _std = t[2].std()
        print("Parameters: {}, Mean: {}, Std: {}".format(_params, _mean, _std))
    print("Best parameter {} with scoring {}".format(g_model.best_params_, g_model.best_score_))
    return g_model


if __name__ == "__main__":
    train_x, train_y, test_x, test_id = get_data()
    train_x_std, test_x_std, _= scale_data(train_x, test_x)
    # 特征筛选
    pearsonr_filter = SelectKBest(k=50)
    filter_train = pearsonr_filter.fit_transform(train_x_std, train_y)
    filter_test = pearsonr_filter.transform(test_x_std)

    # SVM模型
    svm = SVC(C=40, kernel="rbf", gamma="auto", cache_size=1024)
    # params = {"C": [30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1600]}
    params = {"gamma": [0.005, 0.008, 0.01, 0.02, 0.03,0.04, 0.05]}
    grid_search(svm, params, filter_train, train_y)