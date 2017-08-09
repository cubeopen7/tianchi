# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from constant import *
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


random_seed = 420


MISSING_COLUMNS = ['残疾军人医疗补助基金支付金额', '公务员医疗补助基金支付金额', '出院诊断病种名称', '补助审批金额', '城乡救助补助金额',
                   '一次性医用材料申报金额', '民政救助补助金额', '城乡优抚补助金额', '医疗救助医院申请', '本次审批金额']
TIME_COLUMNS= ["交易时间", "住院开始时间", "住院终止时间", "申报受理时间", "操作时间"]
# RAW FEATURE
FLAG_MAX = ["起付线标准金额"]
FLAG_SUM_MAX = ["医疗救助个人按比例负担金额", "最高限额以上金额", "公务员医疗补助基金支付金额", "城乡救助补助金额", "医疗救助医院申请",
                "残疾军人医疗补助基金支付金额", "民政救助补助金额", "城乡优抚补助金额"]
SUM_MEAN_MAX = ["起付标准以上自负比例金额", "基本医疗保险统筹基金支付金额", "可用账户报销金额", "基本医疗保险个人账户支付金额",
                "非账户支付金额", "本次审批金额", "补助审批金额"]
FLAG_SUM_MEAN_MAX = ["药品费发生金额", "贵重药品发生金额", "中成药费发生金额", "中草药费发生金额", "药品费自费金额", "药品费申报金额",
                     "检查费发生金额", "贵重检查费金额", "检查费自费金额", "检查费申报金额", "治疗费发生金额", "治疗费自费金额", "治疗费申报金额",
                     "手术费发生金额", "手术费自费金额", "手术费申报金额", "床位费发生金额", "床位费申报金额", "医用材料发生金额",
                     "高价材料发生金额", "医用材料费自费金额", "成分输血申报金额", "其它发生金额", "其它申报金额", "一次性医用材料申报金额"]
# MADE FEATURE
M_FLAG = ["病症缺失"]
M_MEAN_MIN = ["医保报销比例"]
M_MEAN_NORMAL = ["病症数量"]
M_MEAN_MAX = ["贵重药品比例", "中成药品比例", "中药比例", "药品自费比例", "药品报销比例", "贵重检查费比例", "检查费自费比例", "检查费报销比例",
              "治疗费自费比例", "治疗费报销比例", "手术费自费比例", "手术费报销比例", "床位费报销比例", "贵重比例"]
M_FLAG_SUM_MAX = ["贵重金额"]
M_SUM_MAX_MEAN_NORMAL = ["总费用"]


def missing(df):
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

def get_feature(df):
    # 1. 药品
    df["贵重药品比例"] = df["贵重药品发生金额"] / df["药品费发生金额"]
    df["贵重药品比例"] = df["贵重药品比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["中成药品比例"] = df["中成药费发生金额"] / df["药品费发生金额"]
    df["中成药品比例"] = df["中成药品比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["中药比例"] = df["中草药费发生金额"] / df["药品费发生金额"]
    df["中药比例"] = df["中药比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["药品自费比例"] = df["药品费自费金额"] / df["药品费发生金额"]
    df["药品自费比例"] = df["药品自费比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["药品报销比例"] = df["药品费申报金额"] / df["药品费发生金额"]
    df["药品报销比例"] = df["药品报销比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    # 2. 检查费
    df["贵重检查费比例"] = df["贵重检查费金额"] / df["检查费发生金额"]
    df["贵重检查费比例"] = df["贵重检查费比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["检查费自费比例"] = df["检查费自费金额"] / df["检查费发生金额"]
    df["检查费自费比例"] = df["检查费自费比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["检查费报销比例"] = df["检查费申报金额"] / df["检查费发生金额"]
    df["检查费报销比例"] = df["检查费报销比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    # 3. 治疗费
    df["治疗费自费比例"] = df["治疗费自费金额"] / df["治疗费发生金额"]
    df["治疗费自费比例"] = df["治疗费自费比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["治疗费报销比例"] = df["治疗费申报金额"] / df["治疗费发生金额"]
    df["治疗费报销比例"] = df["治疗费报销比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    # 4. 手术费
    df["手术费自费比例"] = df["手术费自费金额"] / df["手术费发生金额"]
    df["手术费自费比例"] = df["手术费自费比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["手术费报销比例"] = df["手术费申报金额"] / df["手术费发生金额"]
    df["手术费报销比例"] = df["手术费报销比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    # 5. 床位费
    df["床位费报销比例"] = df["床位费申报金额"] / df["床位费发生金额"]
    df["床位费报销比例"] = df["床位费报销比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    # 6. 总体
    df["总费用"] = df["药品费发生金额"] + df["检查费发生金额"] + df["治疗费发生金额"] + df["手术费发生金额"] + df["床位费发生金额"] + df["医用材料发生金额"] + df["其它发生金额"]
    df["总费用"] = df["总费用"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["医保报销比例"] = df["本次审批金额"] / df["总费用"]
    df["医保报销比例"] = df["医保报销比例"].map(lambda x: 0.0 if np.isnan(x) else x)
    df["贵重金额"] = df["贵重药品发生金额"] + df["贵重检查费金额"] + df["高价材料发生金额"]
    df["贵重比例"] = df["贵重金额"] / df["总费用"]
    # 7. 疾病类型
    disease = df["出院诊断病种名称"]
    df["病症缺失"] = disease.isnull().map(lambda x: 1 if x else 0)
    def disease_ana(data):
        if isinstance(data, float):
            return 0
        for s in [";", "，", ",", "；", " "]:
            if s not in data:
                continue
            dises = data.split(s)
            return len(dises)
        print(data)
        return 1
    df["病症数量"] = disease.map(disease_ana)

    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    # train_feature = pd.read_csv("./data/df_train.csv")  # (1830386, 69)
    # test_feature = pd.read_csv("./data/df_test.csv")  # (362607, 69)
    # train_feature = train_feature[[col for col in train_feature.columns if col not in USELESS_COLUMNS]]
    # test_feature = test_feature[[col for col in test_feature.columns if col not in USELESS_COLUMNS]]

    # # 查看每列情况
    # for col in train_feature.columns:
    #     element_count = train_feature[col].unique()
    #     print(element_count)
    #     print("列名称: {}, 列类型: {}, 列值种类数: {}".format(col, train_feature[col].dtypes, len(element_count)))
    # missing(train_feature)

    # # 处理时间列
    # def date_handle(data):
    #     try:
    #         if isinstance(data, float) or isinstance(data, int):
    #             return None
    #         data = data.replace(" ", "")
    #         date = datetime.datetime.strptime(data, "%d-%m月-%y")
    #         return date
    #     except Exception as e:
    #         print(data, type(data), e)
    #         return None
    #
    # for col in TIME_COLUMNS:
    #     if col in ["交易时间"]:
    #         train_feature[col] = train_feature[col].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    #     else:
    #         train_feature[col] = train_feature[col].map(date_handle)
    # train_feature.to_csv("re_test.csv", index=False)
    # train_feature = pd.read_csv("re_train.csv", encoding="gbk")
    # test_feature = pd.read_csv("re_test.csv", encoding="gbk")
    # date_feature = train_feature[TIME_COLUMNS]
    # def is_date_same(data):
    #     data = data.tolist()
    #     t_date = None
    #     for t in data:
    #         if t_date is None:
    #             t_date = t
    #             continue
    #         if not isinstance(t, str):
    #             continue
    #         if t != t_date:
    #             return True
    #     return False
    #
    # b_same = date_feature.apply(is_date_same, axis=1)
    # print(date_feature[b_same])
    # print(date_feature[b_same].shape)
    # # 分析得出这么多种的时间都是重复的, 统一使用无缺失的交易时间作为日期
    # train_feature.drop(["住院开始时间", "住院终止时间", "申报受理时间", "操作时间"], axis=1, inplace=True)
    # test_feature.drop(["住院开始时间", "住院终止时间", "申报受理时间", "操作时间"], axis=1, inplace=True)
    # train_feature.to_csv("re_train1.csv", index=False)
    # test_feature.to_csv("re_test1.csv", index=False)

    # train_feature = pd.read_csv("re_train.csv", encoding="gbk")
    # # test_feature = pd.read_csv("re_test.csv", encoding="gbk")
    # train_label = pd.read_csv("./data/df_id_train.csv", header=None)
    # train_label.columns = ["个人编码", "label"]
    # train = train_feature.merge(train_label, on="个人编码", how="left")
    # train.to_csv("train.csv", index=False)

    # 修补MISSING数据
    # train = pd.read_csv("train.csv", encoding="gbk")
    # test = pd.read_csv("re_test.csv", encoding="gbk")
    # 1. 本次审批金额
    # t = train[train["本次审批金额"].isnull()]
    # t["本次审批金额"] = t["起付标准以上自负比例金额"] + t["基本医疗保险统筹基金支付金额"]
    # train["本次审批金额"][train["本次审批金额"].isnull()] = t["本次审批金额"]
    # t = test[test["本次审批金额"].isnull()]
    # t["本次审批金额"] = t["起付标准以上自负比例金额"] + t["基本医疗保险统筹基金支付金额"]
    # test["本次审批金额"][test["本次审批金额"].isnull()] = t["本次审批金额"]

    # # 造特征
    # get_feature(test)
    # # 用户特征
    def get_max(data, col):
        t_max = data[col].max().apply(lambda x: 0 if np.isnan(x) else x)
        t_max.index = [t + "_MAX" for t in t_max.index.tolist()]
        return t_max

    def get_min(data, col):
        t = data[col].min().apply(lambda x: 0 if np.isnan(x) else x)
        t.index = [s + "_MIN" for s in t.index.tolist()]
        return t

    def get_sum(data, col):
        t_sum = data[col].sum().apply(lambda x: 0 if np.isnan(x) else x)
        t_sum.index = [t + "_SUM" for t in t_sum.index.tolist()]
        return t_sum

    def get_mean(data, col):
        t = data[col].mean().apply(lambda x: 0 if np.isnan(x) else x)
        t.index = [s + "_MEAN" for s in t.index.tolist()]
        return t

    def get_flag(data, col):
        t = (get_max(data, col) > 0.0).apply(lambda x: 1 if x else 0)
        t.index = [s.replace("MAX", "FLAG") for s in t.index.tolist()]
        return t

    def get_manual_mean(data, col):
        cash = data["总费用"]
        total_cash = cash.sum()
        for c in col:
            data[c] = data[c] * cash
        t = data[col].sum() / total_cash
        t.index = [s + "_MEAN" for s in t.index.tolist()]
        return t

    # data_info = [("train_1.csv", "user_train_id.csv"),
    #              ("test_1.csv", "user_test_id.csv")]
    # for read, save in data_info:
    #     data_df = pd.read_csv(read, encoding="gbk")
    #     data_df = data_df.fillna(0.0)
    #     data_group= data_df.groupby("个人编码")
    #     uid_list = []
    #     for i, (uid, data) in enumerate(tqdm(data_group)):
    #         # 1. 最大值-是否存在
    #         t1 = get_max(data, FLAG_MAX)
    #         t2 = get_flag(data, FLAG_MAX)
    #         t = t1
    #         t = t.append(t2)
    #         # 2. 最大值-是否存在-和
    #         t3 = get_max(data, FLAG_SUM_MAX)
    #         t4 = get_sum(data, FLAG_SUM_MAX)
    #         t5 = get_flag(data, FLAG_SUM_MAX)
    #         t = t.append([t3, t4, t5])
    #         # 3. 最大值-均值-和
    #         t6 = get_max(data, SUM_MEAN_MAX)
    #         t7 = get_sum(data, SUM_MEAN_MAX)
    #         t8 = get_mean(data, SUM_MEAN_MAX)
    #         t = t.append([t6, t7, t8])
    #         # 4.是否存在-和-均值-最大值
    #         t9 = get_flag(data, FLAG_SUM_MEAN_MAX)
    #         t10 = get_max(data, FLAG_SUM_MEAN_MAX)
    #         t11 = get_sum(data, FLAG_SUM_MEAN_MAX)
    #         t12 = get_mean(data, FLAG_SUM_MEAN_MAX)
    #         t = t.append([t9, t10, t11, t12])
    #         # 5. 后创特征: 是否存在
    #         t13 = get_flag(data, M_FLAG)
    #         t = t.append(t13)
    #         # 6. 后创特征: 均值-最小值
    #         t14 = get_manual_mean(data, M_MEAN_MIN)
    #         t15 = get_min(data, M_MEAN_MIN)
    #         t = t.append([t14, t15])
    #         # 7. 后创特征: 均值-最大值
    #         t16 = get_manual_mean(data, M_MEAN_MAX)
    #         t17 = get_max(data, M_MEAN_MAX)
    #         t = t.append([t16, t17])
    #         # 8. 后创特征: 正常均值
    #         t18 = get_mean(data, M_MEAN_NORMAL)
    #         t = t.append(t18)
    #         # 9. 后创特征: 是否存在-和-最大值
    #         t19 = get_flag(data, M_FLAG_SUM_MAX)
    #         t20 = get_sum(data, M_FLAG_SUM_MAX)
    #         t21 = get_max(data, M_FLAG_SUM_MAX)
    #         t = t.append([t19, t20, t21])
    #         # 10. 后创特征: 最大值-和-正常均值
    #         t22 = get_mean(data, M_SUM_MAX_MEAN_NORMAL)
    #         t23 = get_max(data, M_SUM_MAX_MEAN_NORMAL)
    #         t24 = get_sum(data, M_SUM_MAX_MEAN_NORMAL)
    #         t = t.append([t22, t23, t24])
    #         # 11. 新特征: 顺序号数量, 顺序号最大时间间隔, 平均每日发生的数量
    #         t25 = data.shape[0]
    #         date_col = data["交易时间"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    #         t26 = (date_col.max() - date_col.min()).days
    #         t26 = t26 if t26 > 0 else 1
    #         t27 = t25 / t26
    #         t = t.append(pd.Series([t25, t26, t27], index=["顺序号数量", "顺序号最大时间间隔", "平均每日发生的数量"]))
    #         # 12.新特征: 历史诊断疾病总数
    #         disease_name_col = data["出院诊断病种名称"]
    #         disease_str = ""
    #         for s in disease_name_col:
    #             if isinstance(s, float):
    #                 continue
    #             disease_str = disease_str + s + ";"
    #         for s in ["，", ",", "；", " "]:
    #             disease_str = disease_str.replace(s, ";")
    #         disease_list = disease_str.split(";")
    #         disease_list = list(set(disease_list))
    #         t = t.append(pd.Series(len(disease_list), index=["历史诊断疾病总数"]))
    #         # 用户ID
    #         t = t.append(pd.Series(uid, index=["UID"]))
    #         # 整理添加
    #         if i == 0:
    #             result_df = pd.DataFrame([list(t)], columns=t.index.tolist())
    #         else:
    #             result_df = result_df.append(t, ignore_index=True)
    #     result_df.to_csv(save, index=False)

    # 新特征
    # # 1. 详细表统计特征
    # detail = pd.read_csv("detail_result.csv", encoding="gbk")
    # train = pd.read_csv("train_1.csv", encoding="gbk")
    # train = train.merge(detail, how="left", on="顺序号")
    # test = pd.read_csv("test_1.csv", encoding="gbk")
    # test = test.merge(detail, how="left", on="顺序号")
    # data_pair = [(train, "user_train_new.csv"), (test, "user_test_new.csv")]
    # columns = ["详细数量", "详细均价", "详细最高价"]
    # for input_data, output_data in data_pair:
    #     group = input_data.groupby("个人编码")
    #     for i, (uid, data) in enumerate(tqdm(group)):
    #         t = pd.Series(uid, index=["UID"])
    #         # # 1. 均值-最大值-最小值
    #         # t1 = get_max(data, columns)
    #         # t2 = get_min(data, columns)
    #         # t3 = get_mean(data, columns)
    #         # t = t.append([t1, t2, t3])
    #         # # 2. 是否有隔天单行为
    #         # t4 = get_flag(data, ["详细间隔日期"])
    #         # t = t.append(t4)
    #         # 2. 医院数量
    #         hos_group = data[["个人编码", "医院编码"]].groupby("医院编码").count()
    #         t5 = pd.Series(hos_group.shape[0], index=["就诊医院数量"])
    #         t6 = pd.Series(hos_group["个人编码"].max() / data.shape[0], index=["最常去医院就诊比例"])
    #         t7 = pd.Series(hos_group["个人编码"].min() / data.shape[0], index=["最少去医院就诊比例"])
    #         t8 = hos_group["个人编码"].std() / data.shape[0]
    #         t8 = pd.Series(0.0 if np.isnan(t8) else t8, index=["医院就诊次数标准差"])
    #         t = t.append([t5, t6, t7, t8])
    #         if i == 0:
    #             result_df = pd.DataFrame([list(t)], columns=t.index.tolist())
    #         else:
    #             result_df = result_df.append(t, ignore_index=True)
    #     result_df["UID"] = result_df["UID"].astype(np.int64)
    #     result_df.to_csv(output_data, index=False)

    # train_1 = pd.read_csv("user_train.csv", encoding="gbk")
    # train_2 = pd.read_csv("user_train_new.csv", encoding="gbk")
    # train_id = pd.read_csv("user_train_id.csv", encoding="gbk", header=None)
    # train_id.columns = ["UID"]
    # train = pd.concat([train_id, train_1], axis=1)
    # train = train.merge(train_2, how="left", on="UID")
    # train.to_csv("train_x.csv", index=False)

    # test_1 = pd.read_csv("user_test.csv", encoding="gbk")
    # test_2 = pd.read_csv("user_test_new.csv", encoding="gbk")
    # test_id = pd.read_csv("user_test_id.csv", encoding="gbk", header=None)
    # test_id.columns = ["UID"]
    # test = pd.concat([test_id, test_1], axis=1)
    # test = test.merge(test_2, how="left", on="UID")
    # test.to_csv("test_x.csv", index=False)

    # train_1 = pd.read_csv("train_x.csv", encoding="gbk")
    # train_2 = pd.read_csv("user_train_new.csv", encoding="gbk")
    # train_1 = train_1.merge(train_2, how="left", on="UID")
    # train_1.to_csv("train_x_2.csv", index=False)
    #
    # test_1 = pd.read_csv("test_x.csv", encoding="gbk")
    # test_2 = pd.read_csv("user_test_new.csv", encoding="gbk")
    # test_1 = test_1.merge(test_2, how="left", on="UID")
    # test_1.to_csv("test_x_2.csv", index=False)

    # 消除inf异常值, 并清除常数列(特征只有一种数据, 无效特征)
    train = pd.read_csv("train_x_2.csv", encoding="gbk")
    test = pd.read_csv("test_x_2.csv", encoding="gbk")
    train = train.replace(np.inf, 0)
    test = test.replace(np.inf, 0)
    for col in train.columns:
        if len(train[col].unique()) < 2:
            train.drop(col, axis=1, inplace=True)
            test.drop(col, axis=1, inplace=True)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

    # # 合并疾病详细数据
    # train = pd.read_csv("train.csv", encoding="gbk")
    # test = pd.read_csv("test.csv", encoding="gbk")
    # dis_train = pd.read_csv("disease_train.csv", encoding="gbk")
    # dis_test = pd.read_csv("disease_test.csv", encoding="gbk")
    # train = pd.concat([train, dis_train], axis=1)
    # test = pd.concat([test, dis_test], axis=1)
    # train.to_csv("train.csv", index=False)
    # test.to_csv("test.csv", index=False)

    # # 标准化处理, 并进行降维, 将降维数据作为新的特征
    # train = pd.read_csv("train.csv", encoding="gbk")
    # train_id, train = train[["UID"]], train.drop("UID", axis=1)
    # test = pd.read_csv("test.csv", encoding="gbk")
    # test_id, test = test[["UID"]], test.drop("UID", axis=1)
    # std_model = StandardScaler()
    # train_std = std_model.fit_transform(train)
    # test_std = std_model.transform(test)
    # pca_model = PCA(n_components=25, random_state=random_seed)
    # train_pca = pca_model.fit_transform(train_std)
    # test_pca = pca_model.transform(test_std)
    # train_pca = pd.DataFrame(train_pca, columns=["PCA_" + str(i) for i in range(pca_model.n_components)])
    # test_pca = pd.DataFrame(test_pca, columns=["PCA_" + str(i) for i in range(pca_model.n_components)])
    # ica_model = FastICA(n_components=25, random_state=random_seed)
    # train_ica = ica_model.fit_transform(train_std)
    # test_ica = ica_model.transform(test_std)
    # train_ica = pd.DataFrame(train_ica, columns=["ICA_" + str(i) for i in range(pca_model.n_components)])
    # test_ica = pd.DataFrame(test_ica, columns=["ICA_" + str(i) for i in range(pca_model.n_components)])
    # train = pd.concat([train_id, train, train_pca, train_ica], axis=1)
    # test = pd.concat([test_id, test, test_pca, test_ica], axis=1)
    # train.to_csv("train.csv", index=False)
    # test.to_csv("test.csv", index=False)