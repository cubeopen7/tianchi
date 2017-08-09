# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from constant import USELESS_COLUMNS


RATIO_FEATURES = ["贵重药品比例", "中成药品比例", "中药比例", "药品自费比例", "药品报销比例",
                  "贵重检查费比例", "检查费自费比例", "检查费报销比例",
                  "治疗费自费比例", "治疗费报销比例",
                  "手术费自费比例", "手术费报销比例",
                  "床位费报销比例",
                  "高价材料比例", "一次性医用材料申报比例", "医用材料自费比例",
                  "报销比例", "贵重比例", "医保基金比例", "个人额外比例", "个人医保自付比例"]
SERIOUS_DISEASE = ["尿毒", "透析", "衰竭", "硬化", "白血病", "死", "梗", "癌", "瘫", "瘤"]
MISSING_COLUMNS = ['残疾军人医疗补助基金支付金额', '公务员医疗补助基金支付金额', '出院诊断病种名称', '补助审批金额', '城乡救助补助金额', '一次性医用材料申报金额', '民政救助补助金额', '城乡优抚补助金额', '医疗救助医院申请']
DISEASE_TYPE = ["糖尿病", "心脏疾病", "脑部疾病", "肾部疾病", "肝部疾病", "肺部疾病", "胃部疾病", "骨科疾病", "免疫疾病", "血管疾病", "癌症症状", "神经疾病", "精神疾病"]
HANDLE_DICT = {
    "min": ["起付线标准金额"],
    "max": ["药品费发生金额", "贵重药品发生金额", "中成药费发生金额", "中草药费发生金额", "药品费自费金额", "药品费申报金额",
             "检查费发生金额", "贵重检查费金额", "检查费自费金额", "检查费申报金额", "治疗费发生金额", "治疗费自费金额", "治疗费申报金额",
             "手术费发生金额", "手术费自费金额", "手术费申报金额", "床位费发生金额", "床位费申报金额",
             "医用材料发生金额", "高价材料发生金额", "医用材料费自费金额", "一次性医用材料申报金额",
             "成分输血申报金额", "其它发生金额", "其它申报金额", "起付线标准金额", "起付标准以上自负比例金额", "基本医疗保险统筹基金支付金额",
             "医疗救助个人按比例负担金额", "最高限额以上金额", "可用账户报销金额", "基本医疗保险个人账户支付金额", "非账户支付金额",
             "公务员医疗补助基金支付金额", "城乡救助补助金额", "补助审批金额", "医疗救助医院申请", "残疾军人医疗补助基金支付金额",
             "民政救助补助金额", "城乡优抚补助金额", "本次审批金额", "总费用", "贵重金额", "个人额外费用", "个人医保自付费用", "病症数量",
             "疾病大类数量"],
    "sum": ["药品费发生金额", "贵重药品发生金额", "中成药费发生金额", "中草药费发生金额", "药品费自费金额", "药品费申报金额",
             "检查费发生金额", "贵重检查费金额", "检查费自费金额", "检查费申报金额", "治疗费发生金额", "治疗费自费金额", "治疗费申报金额",
             "手术费发生金额", "手术费自费金额", "手术费申报金额", "床位费发生金额", "床位费申报金额",
             "医用材料发生金额", "高价材料发生金额", "医用材料费自费金额", "一次性医用材料申报金额",
             "成分输血申报金额", "其它发生金额", "其它申报金额", "起付标准以上自负比例金额", "基本医疗保险统筹基金支付金额",
             "医疗救助个人按比例负担金额", "最高限额以上金额", "可用账户报销金额", "基本医疗保险个人账户支付金额", "非账户支付金额",
             "公务员医疗补助基金支付金额", "城乡救助补助金额", "补助审批金额", "医疗救助医院申请", "残疾军人医疗补助基金支付金额",
             "民政救助补助金额", "城乡优抚补助金额", "本次审批金额", "总费用", "贵重金额", "个人额外费用", "个人医保自付费用"],
    "mean": ["药品费发生金额", "贵重药品发生金额", "中成药费发生金额", "中草药费发生金额", "药品费自费金额", "药品费申报金额",
              "检查费发生金额", "贵重检查费金额", "检查费自费金额", "检查费申报金额", "治疗费发生金额", "治疗费自费金额", "治疗费申报金额",
              "手术费发生金额", "手术费自费金额", "手术费申报金额", "床位费发生金额", "床位费申报金额",
              "医用材料发生金额", "高价材料发生金额", "医用材料费自费金额", "一次性医用材料申报金额",
              "成分输血申报金额", "其它发生金额", "其它申报金额", "起付标准以上自负比例金额", "基本医疗保险统筹基金支付金额",
              "可用账户报销金额", "基本医疗保险个人账户支付金额", "非账户支付金额", "公务员医疗补助基金支付金额", "城乡救助补助金额", "补助审批金额",
              "医疗救助医院申请", "残疾军人医疗补助基金支付金额", "民政救助补助金额", "城乡优抚补助金额", "本次审批金额", "总费用", "贵重金额",
              "个人额外费用", "个人医保自付费用", "病症数量", "疾病大类数量"],
    "flag": ["药品费发生金额", "贵重药品发生金额", "中成药费发生金额", "中草药费发生金额", "药品费自费金额", "药品费申报金额",
              "检查费发生金额", "贵重检查费金额", "检查费自费金额", "检查费申报金额", "治疗费发生金额", "治疗费自费金额", "治疗费申报金额",
              "手术费发生金额", "手术费自费金额", "手术费申报金额", "床位费发生金额", "床位费申报金额",
              "医用材料发生金额", "高价材料发生金额", "医用材料费自费金额", "一次性医用材料申报金额",
              "成分输血申报金额", "其它发生金额", "其它申报金额", "起付线标准金额", "医疗救助个人按比例负担金额", "最高限额以上金额",
              "可用账户报销金额", "基本医疗保险个人账户支付金额", "非账户支付金额", "公务员医疗补助基金支付金额", "城乡救助补助金额", "补助审批金额",
              "医疗救助医院申请", "残疾军人医疗补助基金支付金额", "民政救助补助金额", "城乡优抚补助金额", "贵重金额", "个人额外费用"],
}
# 外挂数据
# # 1. 医院数据
# hos_details = pd.read_csv("./hosp_freq.csv", encoding="gbk")
# hos_details["医院编号"] = hos_details["医院编号"].astype(int)
# hos_details["假出现次数"] = hos_details["假出现次数"].astype(int)
# hos_details.loc[hos_details["假出现次数"] + hos_details["真出现次数"] <= 100, "频率"] = 0.0
# hos_details = hos_details[["医院编号", "频率"]].set_index("医院编号")
# hos_details = hos_details["频率"]


def missing(data):
    print("数据总大小为: {}".format(data.shape))
    data_shape = data.shape
    data_samples = data_shape[0]
    data_columns = data_shape[1]
    missing_data = data.isnull()
    missing_columns = missing_data.sum(axis=0)
    missing_columns = missing_columns[missing_columns > 0].sort_values(ascending=False)
    if len(missing_columns) == 0:
        print("没有任何缺失数据")
        return
    else:
        missing_col_count = len(missing_columns)
        print("有{0}列数据缺失，占总列数的{1}%".format(missing_col_count, round(missing_col_count/data_columns, 6) * 100))
        print("缺失的列为: {}".format(missing_columns.index.values))
    missing_samples = missing_data.sum(axis=1)
    missing_samples = missing_samples[missing_samples > 0]
    missing_spl_count = len(missing_samples)
    print("有{0}行数据缺失，占总行数的{1}%".format(missing_spl_count, round(missing_spl_count/data_samples, 6) * 100))
    return


def get_feature(data):
    # 1. 药品
    data["贵重药品比例"] = data["贵重药品发生金额"] / data["药品费发生金额"]
    data["中成药品比例"] = data["中成药费发生金额"] / data["药品费发生金额"]
    data["中药比例"] = data["中草药费发生金额"] / data["药品费发生金额"]
    data["药品自费比例"] = data["药品费自费金额"] / data["药品费发生金额"]
    data["药品报销比例"] = data["药品费申报金额"] / data["药品费发生金额"]
    # 2. 检查费
    data["贵重检查费比例"] = data["贵重检查费金额"] / data["检查费发生金额"]
    data["检查费自费比例"] = data["检查费自费金额"] / data["检查费发生金额"]
    data["检查费报销比例"] = data["检查费申报金额"] / data["检查费发生金额"]
    # 3. 治疗费
    data["治疗费自费比例"] = data["治疗费自费金额"] / data["治疗费发生金额"]
    data["治疗费报销比例"] = data["治疗费申报金额"] / data["治疗费发生金额"]
    # 4. 手术费
    data["手术费自费比例"] = data["手术费自费金额"] / data["手术费发生金额"]
    data["手术费报销比例"] = data["手术费申报金额"] / data["手术费发生金额"]
    # 5. 床位费
    data["床位费报销比例"] = data["床位费申报金额"] / data["床位费发生金额"]
    # 6. 医用材料费
    data["高价材料比例"] = data["高价材料发生金额"] / data["医用材料发生金额"]
    data["一次性医用材料申报比例"] = data["一次性医用材料申报金额"] / data["医用材料发生金额"]
    data["医用材料自费比例"] = data["医用材料费自费金额"] / data["医用材料发生金额"]
    # 7. 综合
    data["总费用"] = data["药品费发生金额"] + data["检查费发生金额"] + data["治疗费发生金额"] + data["手术费发生金额"] + data["床位费发生金额"] + data["医用材料发生金额"] + data["其它发生金额"]
    data["贵重金额"] = data["贵重药品发生金额"] + data["贵重检查费金额"] + data["高价材料发生金额"]
    data["报销比例"] = data["本次审批金额"] / data["总费用"]
    data["贵重比例"] = data["贵重金额"] / data["总费用"]
    data["医保基金比例"] = data["基本医疗保险统筹基金支付金额"] / data["本次审批金额"]
    data["个人额外费用"] = (data["总费用"] - data["本次审批金额"]).map(lambda x:x if x > 0 else 0)
    data["个人医保自付费用"] = data["个人额外费用"] + data["起付标准以上自负比例金额"]
    data["个人额外比例"] = data["个人额外费用"] / data["总费用"]
    data["个人医保自付比例"] = data["个人医保自付费用"] / data["总费用"]
    # NAN值补零
    data = data.fillna(0.0)
    # 8. 出院诊断病种名称转换
    for c in  ["，", ",", "；", ".", "、", "@", " ", "\t", ":", "："]:
        data["出院诊断病种名称"] = data["出院诊断病种名称"].str.replace(c, ";")
    def list2str(data):
        if isinstance(data, float):
            return ""
        res = ""
        for t in data:
            if len(t) <= 1:
                continue
            res = res + t + ";"
        if len(res) == 0:
            return res
        else:
            return res[:-1]
    data["出院诊断病种名称"] = data["出院诊断病种名称"].str.split(";").map(list2str)
    data["病症数量"] = data["出院诊断病种名称"].map(lambda x:0 if isinstance(x, float) else len(x.split(";")))
    data["疾病大类数量"] = data[DISEASE_TYPE].sum(axis=1)
    return data


def get_user_feature(raw_data):
    def _max(data, col):
        t_max = data[col].max()
        t_max.index = [t + "_max" for t in t_max.index.tolist()]
        return t_max
    def _min(data, col):
        t = data[col].min()
        t.index = [s + "_min" for s in t.index.tolist()]
        return t
    def _sum(data, col):
        t_sum = data[col].sum()
        t_sum.index = [t + "_sum" for t in t_sum.index.tolist()]
        return t_sum
    def _mean(data, col):
        t = data[col].mean()
        t.index = [s + "_mean" for s in t.index.tolist()]
        return t
    def _flag(data, col):
        t = (_max(data, col) > 0.00000001).apply(lambda x: 1 if x else 0)
        t.index = [s.replace("max", "flag") for s in t.index.tolist()]
        return t
    _func_dict = {
        "max": _max,
        "min": _min,
        "sum": _sum,
        "mean": _mean,
        "flag": _flag,
    }
    def _mean_std(data, mean_method="median"):
        if len(data) == 0:
            return 0.0, 0.0
        if mean_method == "median":
            _mean = data.median()
        else:
            _mean = data.mean()
        data = data / _mean
        _std = data.std()
        return _mean, _std

    data_group = raw_data.groupby("个人编码")
    for i, (uid, data) in enumerate(tqdm(data_group)):
        data["交易时间"] = data["交易时间"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
        data = data.sort_values("交易时间")
        # 1. UID
        t = pd.Series(uid, index=["uid"])
        # 2. 数值型数据统计
        for k, v in HANDLE_DICT.items():
            t = t.append(_func_dict[k](data, v))
        # 3. 比例型数据统计
        t = t.append(_max(data, RATIO_FEATURES))
        for col in RATIO_FEATURES:
            try:
                if "药" in col:
                    t_cash = data[col] * data["药品费发生金额"]
                    t_mean = t_cash.sum() / data["药品费发生金额"].sum()
                elif "检查" in col:
                    t_cash = data[col] * data["检查费发生金额"]
                    t_mean = t_cash.sum() / data["检查费发生金额"].sum()
                elif "治疗" in col:
                    t_cash = data[col] * data["治疗费发生金额"]
                    t_mean = t_cash.sum() / data["治疗费发生金额"].sum()
                elif "手术" in col:
                    t_cash = data[col] * data["手术费发生金额"]
                    t_mean = t_cash.sum() / data["手术费发生金额"].sum()
                elif "床位" in col:
                    t_cash = data[col] * data["床位费发生金额"]
                    t_mean = t_cash.sum() / data["床位费发生金额"].sum()
                elif "材料" in col:
                    t_cash = data[col] * data["医用材料发生金额"]
                    t_mean = t_cash.sum() / data["医用材料发生金额"].sum()
                else:
                    t_cash = data[col] * data["总费用"]
                    t_mean = t_cash.sum() / data["总费用"].sum()
            except ZeroDivisionError as e:
                t_mean = 0.0
            t = t.append(pd.Series(t_mean, index=[col + "_mean"]))
        # 4. 其他特征
        t4_1 = pd.Series(len(data), index=["单号总数"])
        t = t.append(t4_1)
        # 5. 金额特征
        t5_1 = pd.Series(data["药品费发生金额"].sum() / data["总费用"].sum(), index=["药品金额比例"])
        t5_2 = pd.Series(data["治疗费发生金额"].sum() / data["总费用"].sum(), index=["治疗金额比例"])
        t_series_1 = data["药品费发生金额"][data["药品费发生金额"] > 100]
        t_median_1, t_std_1 = _mean_std(t_series_1)
        t5_3 = pd.Series(t_median_1, index=["大额药品金额中位数"])
        t5_4 = pd.Series(t_std_1, index=["大额药品金额标准差"])
        t_series_2 = data["治疗费发生金额"][data["治疗费发生金额"] > 100]
        t_median_2, t_std_2 = _mean_std(t_series_2)
        t5_5 = pd.Series(t_median_2, index=["大额治疗费金额中位数"])
        t5_6 = pd.Series(t_std_2, index=["大额治疗费金额标准差"])
        t_series_3 = data[["总费用", "基本医疗保险统筹基金支付金额"]][data["总费用"] > 100]
        t5_7 = pd.Series(len(t_series_3) / len(data), index=["大额费用比例"])
        t = t.append([t5_1, t5_2, t5_3, t5_4, t5_5, t5_6, t5_7])
        # 6. 日期特征
        data_series = data["交易时间"]
        total_days = (data_series.max() - data_series.min()).days + 1
        count_series = data_series.value_counts().sort_index()
        count_days = len(count_series)
        max_per_day = count_series.max()
        min_per_day = count_series.min()
        if count_days > 1:
            day_series = pd.Series(count_series.index)
            day_delta = (day_series.iloc[1:].reset_index(drop=True) - day_series.iloc[:-1].reset_index(drop=True)).map(lambda x: x.days)
            max_delta = day_delta.max()
            std_delta = count_series.std()
            mean_delta = day_delta.mean()
        else:
            max_delta = 0
            std_delta = 0.0
            mean_delta = 0.0
        t6_1 = pd.Series(len(data) / total_days, index=["平均每天单据数量"])
        t6_2 = pd.Series(len(data) / count_days, index=["平均每次单据数量"])
        t6_3 = pd.Series(total_days, index=["日期跨越天数"])
        t6_4 = pd.Series(count_days, index=["产生交易的日期计数"])
        t6_5 = pd.Series(max_per_day, index=["一天产生的单据最大数量"])
        t6_6 = pd.Series(min_per_day, index=["一天产生的单据最小数量"])
        t6_7 = pd.Series(max_delta, index=["最大间隔天数"])
        t6_8 = pd.Series(mean_delta, index=["平均间隔天数"])
        t6_9 = pd.Series(std_delta, index=["间隔天数标准差"])
        t = t.append([t6_1, t6_2, t6_3, t6_4, t6_5, t6_6, t6_7, t6_8, t6_9])
        # 7. 病情特征
        dis_series = data["出院诊断病种名称"][data["出院诊断病种名称"].notnull()]
        dis_type_df = data[["挂号", "门特挂号"] + DISEASE_TYPE + ["重症疾病"]][data["出院诊断病种名称"].notnull()]
        type_sum = dis_type_df.sum()
        t7_1 = type_sum[["挂号", "门特挂号", "重症疾病"]]
        t7_1.index = [s + "_sum" for s in t7_1.index]
        t7_2 = t7_1 / len(dis_series)
        t7_2.index = [s + "_ratio" for s in t7_2.index]
        t7_3 = type_sum.map(lambda x: 1 if x > 0 else 0)
        t7_3.index = [s + "_flag" for s in t7_3.index]
        t7_4 = pd.Series(t7_3.sum(), index=["疾病类别总数量"])
        dis_total_str = ""
        for dis_single_str in dis_series:
            dis_total_str = dis_total_str + dis_single_str + ";"
        dis_total_str = dis_total_str[:-1]
        dis_list = set(dis_total_str.split(";"))
        t7_5 = pd.Series(len(dis_list), index=["病情总数量"])
        b7_1 = dis_series.map(lambda x: len(x.split(";")) > 1)
        dis_count = dis_series[b7_1].value_counts()
        dis_count = dis_count[dis_count > 1]
        if len(dis_count) == 0:
            repeat_count = repeat_max = repeat_mean = day_delta_max = 0
        else:
            repeat_count = len(dis_count)
            repeat_max = dis_count.max()
            repeat_mean = dis_count.mean()
            dis_df = data[["出院诊断病种名称", "交易时间"]][data["出院诊断病种名称"].notnull()]
            dis_df = dis_df[dis_df["出院诊断病种名称"].isin(list(dis_count.index.tolist()))]
            dis_df["交易时间"] = dis_df["交易时间"]
            dis_df_group = dis_df.groupby("出院诊断病种名称").apply(lambda x: (x["交易时间"].max() - x["交易时间"].min()).days)
            day_delta_max = dis_df_group.max()
        t7_6 = pd.Series([repeat_count, repeat_max, repeat_mean, day_delta_max], index=["重复病情字符串数量", "单病情字符串重复最大值", "单病情字符串重复均值", "重复病情字符串最大日期间隔"])
        t = t.append([t7_1, t7_2, t7_3, t7_4, t7_5, t7_6])
        # 8. 医院特征
        hos_df = data[["医院编码", "交易时间"]].reset_index(drop=True)
        hos_group = hos_df.groupby("医院编码")
        hos_series_1 = hos_group["交易时间"].size()
        hos_count = len(hos_series_1)
        if hos_count <= 1:
            hos_std = hos_max_day_delta = period_other_hos_count = 0
            hos_max_ratio = 1
        else:
            hos_series_2 = hos_group["交易时间"].max() - hos_group["交易时间"].min()
            hos_std = (hos_series_1 / len(data)).std()
            hos_max_day_delta = hos_series_2.max().days
            hos_max_ratio = hos_series_1.max() / len(data)
            if hos_series_1.max() <= 1:
                period_other_hos_count = 0
            else:
                hos_most = hos_series_1.sort_values(ascending=False).index.tolist()[0]
                hos_most_index = hos_df[hos_df["医院编码"] == hos_most].index.tolist()
                _begin, _end = hos_most_index[0], hos_most_index[-1] + 1
                most_period = hos_df["医院编码"].iloc[_begin: _end]
                period_other_hos_count = len(most_period.unique()) - 1
        t8_1 = pd.Series([hos_count, hos_max_ratio, hos_std, hos_max_day_delta, period_other_hos_count], index=["医院总数量", "最常就诊医院占比", "医院次数标准差", "最大相同医院间隔天数", "同医院同时就诊最大医院数量"])
        try:
            hos_factor = pd.factorize(hos_df["医院编码"])[0] + 1
            hos_diff = hos_factor[1:] - hos_factor[:-1]
            hos_diff[0] = 0
            t8_2 = pd.Series(hos_diff.std(), index=["医院梯度标准差"])
        except Exception as e:
            print(e)
            print(data)
            t8_2 = pd.Series(0.0, index=["医院梯度标准差"])
        hos_series = data["医院编码"].value_counts() / len(data)
        hos_weight = hos_details.loc[hos_series.index.tolist()]
        t8_3 = pd.Series((hos_series * hos_weight).sum(), index=["骗保权值"])
        t = t.append([t8_1, t8_2, t8_3])
        if i == 0:
            result_df = pd.DataFrame([list(t)], columns=t.index.tolist())
        else:
            result_df = result_df.append(t, ignore_index=True)
    return result_df


if __name__ == "__main__":
    # # 移除原始数据中的无用列
    # train_raw = pd.read_csv("./data/data_train.csv")
    # test_raw = pd.read_csv("./data/data_test.csv")
    # train_drop = train_raw[[col for col in train_raw.columns if col not in USELESS_COLUMNS]]
    # test_drop = test_raw[[col for col in test_raw.columns if col not in USELESS_COLUMNS]]
    # train_drop.to_csv("./data_tmp/train1.csv", index=False)
    # test_drop.to_csv("./data_tmp/test1.csv", index=False)

    # # 处理时间, 标准化为datetime.datetime格式
    # train_feature = pd.read_csv("./data_tmp/train1.csv", encoding="gbk")
    # test_feature = pd.read_csv("./data_tmp/test1.csv", encoding="gbk")
    # train_feature["交易时间"] = train_feature["交易时间"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    # test_feature["交易时间"] = test_feature["交易时间"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    # train_feature.to_csv("./data_tmp/train2.csv", index=False)
    # test_feature.to_csv("./data_tmp/test2.csv", index=False)

    # # 填补缺失值 - 除去"出院诊断病种名称"字段, 直接全部填写0
    # train_feature = pd.read_csv("./data_tmp/train2.csv", encoding="gbk")
    # test_feature = pd.read_csv("./data_tmp/test2.csv", encoding="gbk")
    # missing(train_feature)
    # missing(test_feature)
    # # 1. 本次审批金额字段
    # t = train_feature[train_feature["本次审批金额"].isnull()]
    # train_feature["本次审批金额"][train_feature["本次审批金额"].isnull()] = t["起付标准以上自负比例金额"] + t["基本医疗保险统筹基金支付金额"]
    # t = test_feature[test_feature["本次审批金额"].isnull()]
    # test_feature["本次审批金额"][test_feature["本次审批金额"].isnull()] = t["起付标准以上自负比例金额"] + t["基本医疗保险统筹基金支付金额"]
    # # 2. 其他字段
    # for col in MISSING_COLUMNS:
    #     if col == "出院诊断病种名称":
    #         continue
    #     train_feature[col].fillna(0.0, inplace=True)
    #     test_feature[col].fillna(0.0, inplace=True)
    # train_feature.to_csv("./data_tmp/train3.csv", index=False)
    # test_feature.to_csv("./data_tmp/test3.csv", index=False)

    # # 处理出院诊断病种名称字段
    # train_feature = pd.read_csv("./data_tmp/train3.csv", encoding="gbk")
    # test_feature = pd.read_csv("./data_tmp/test3.csv", encoding="gbk")
    # # 分析部分
    # total_disease = train_feature["出院诊断病种名称"].append(test_feature["出院诊断病种名称"]).reset_index(drop=True)
    # total_disease = total_disease.fillna("")
    # for c in  ["，", ",", "；", ".", "、", "@", " "]:
    #     total_disease = total_disease.str.replace(c, ";")
    # total_disease = total_disease.str.split(";")
    # total_dis_dict = {}
    # for i in tqdm(range(len(total_disease))):
    #     t = total_disease.iloc[i]
    #     for s in t:
    #         total_dis_dict[s] = total_dis_dict.get(s, 0) + 1
    # dis_stat = pd.Series(total_dis_dict).sort_values(ascending=False)
    # dis_stat.to_csv("dis_stat.csv")
    # dis_stat = pd.read_csv("dis_stat.csv", encoding="gbk")
    # dis_name_col = dis_stat.iloc[:, 0]
    # t1 = dis_stat[-dis_name_col.str.contains("糖尿病")]
    # t2 = dis_stat[dis_name_col.str.contains("心")]
    # t3 = dis_stat[dis_name_col.str.contains("肾")]
    # t4 = dis_stat[(dis_name_col.str.contains("尿")) & (-dis_name_col.str.contains("糖尿病"))]
    # t5 = dis_stat[(dis_name_col.str.contains("透析"))]
    # t6 = dis_stat[(dis_name_col.str.contains("衰竭"))]
    # t7 = dis_stat[(dis_name_col.str.contains("硬化"))]
    # t8 = dis_stat[(dis_name_col.str.contains("动脉"))]
    # t9 = dis_stat[(dis_name_col.str.contains("急性"))]
    # t10 = dis_stat[(dis_name_col.str.contains("死"))]
    # t11 = dis_stat[(dis_name_col.str.contains("脑"))]
    # t12 = dis_stat[(dis_name_col.str.contains("梗"))]
    # t13 = dis_stat[(dis_name_col.str.contains("免疫"))]
    # t14 = dis_stat[(dis_name_col.str.contains("癌"))]
    # t15 = dis_stat[(dis_name_col.str.contains("瘫"))]
    # t16 = dis_stat[(dis_name_col.str.contains("肝"))]
    # t17 = dis_stat[(dis_name_col.str.contains("肿瘤"))]
    # t18 = dis_stat[(dis_name_col.str.contains("胃"))]
    # t19 = dis_stat[(dis_name_col.str.contains("骨"))]
    # t20 = dis_stat[(dis_name_col.str.contains("眼"))]
    # t21 = dis_stat[(dis_name_col.str.contains("耳"))]
    # t22 = dis_stat[(dis_name_col.str.contains("中医"))]
    # t23 = dis_stat[(dis_name_col.str.contains("血管"))]
    # t24 = dis_stat[(dis_name_col.str.contains("神经"))]
    # t25 = dis_stat[(dis_name_col.str.contains("精神"))]
    # t26 = dis_stat[(dis_name_col.str.contains("失眠"))]
    # t27 = dis_stat[(dis_name_col.str.contains("瘤"))]
    # # 生成特征
    # train_feature["出院诊断病种名称"] = train_feature["出院诊断病种名称"].fillna("")
    # train_feature["挂号"] = (train_feature["出院诊断病种名称"].str.contains("挂号") & -train_feature["出院诊断病种名称"].str.contains("门特挂号")).astype(int)
    # train_feature["门特挂号"] = train_feature["出院诊断病种名称"].str.contains("门特挂号").astype(int)
    # train_feature["糖尿病"] = train_feature["出院诊断病种名称"].str.contains("糖尿病").astype(int)
    # train_feature["心脏疾病"] = train_feature["出院诊断病种名称"].str.contains("心").astype(int)
    # train_feature["脑部疾病"] = train_feature["出院诊断病种名称"].str.contains("脑").astype(int)
    # train_feature["肾部疾病"] = (train_feature["出院诊断病种名称"].str.contains("肾") | (train_feature["出院诊断病种名称"].str.contains("尿") & -train_feature["出院诊断病种名称"].str.contains("糖尿"))).astype(int)
    # train_feature["肝部疾病"] = train_feature["出院诊断病种名称"].str.contains("肝").astype(int)
    # train_feature["肺部疾病"] = train_feature["出院诊断病种名称"].str.contains("肺").astype(int)
    # train_feature["胃部疾病"] = train_feature["出院诊断病种名称"].str.contains("胃").astype(int)
    # train_feature["骨科疾病"] = train_feature["出院诊断病种名称"].str.contains("骨").astype(int)
    # train_feature["免疫疾病"] = train_feature["出院诊断病种名称"].str.contains("免疫").astype(int)
    # train_feature["血管疾病"] = (train_feature["出院诊断病种名称"].str.contains("血管") | train_feature["出院诊断病种名称"].str.contains("动脉")).astype(int)
    # train_feature["癌症症状"] = (train_feature["出院诊断病种名称"].str.contains("癌") | train_feature["出院诊断病种名称"].str.contains("白血病") | train_feature["出院诊断病种名称"].str.contains("瘤")).astype(int)
    # train_feature["神经疾病"] = (train_feature["出院诊断病种名称"].str.contains("神经") | train_feature["出院诊断病种名称"].str.contains("癫痫") | train_feature["出院诊断病种名称"].str.contains("中风")).astype(int)
    # train_feature["精神疾病"] = (train_feature["出院诊断病种名称"].str.contains("精神") | train_feature["出院诊断病种名称"].str.contains("抑郁") | train_feature["出院诊断病种名称"].str.contains("失眠") | train_feature["出院诊断病种名称"].str.contains("焦虑")).astype(int)
    # train_feature["重症疾病"] = 0
    # test_feature["出院诊断病种名称"] = test_feature["出院诊断病种名称"].fillna("")
    # test_feature["挂号"] = (test_feature["出院诊断病种名称"].str.contains("挂号") & -test_feature["出院诊断病种名称"].str.contains("门特挂号")).astype(int)
    # test_feature["门特挂号"] = test_feature["出院诊断病种名称"].str.contains("门特挂号").astype(int)
    # test_feature["糖尿病"] = test_feature["出院诊断病种名称"].str.contains("糖尿病").astype(int)
    # test_feature["心脏疾病"] = test_feature["出院诊断病种名称"].str.contains("心").astype(int)
    # test_feature["脑部疾病"] = test_feature["出院诊断病种名称"].str.contains("脑").astype(int)
    # test_feature["肾部疾病"] = (test_feature["出院诊断病种名称"].str.contains("肾") | (
    # test_feature["出院诊断病种名称"].str.contains("尿") & -test_feature["出院诊断病种名称"].str.contains("糖尿"))).astype(int)
    # test_feature["肝部疾病"] = test_feature["出院诊断病种名称"].str.contains("肝").astype(int)
    # test_feature["肺部疾病"] = test_feature["出院诊断病种名称"].str.contains("肺").astype(int)
    # test_feature["胃部疾病"] = test_feature["出院诊断病种名称"].str.contains("胃").astype(int)
    # test_feature["骨科疾病"] = test_feature["出院诊断病种名称"].str.contains("骨").astype(int)
    # test_feature["免疫疾病"] = test_feature["出院诊断病种名称"].str.contains("免疫").astype(int)
    # test_feature["血管疾病"] = (test_feature["出院诊断病种名称"].str.contains("血管") | test_feature["出院诊断病种名称"].str.contains("动脉")).astype(int)
    # test_feature["癌症症状"] = (test_feature["出院诊断病种名称"].str.contains("癌") | test_feature["出院诊断病种名称"].str.contains("白血病") | test_feature["出院诊断病种名称"].str.contains("瘤")).astype(int)
    # test_feature["神经疾病"] = (test_feature["出院诊断病种名称"].str.contains("神经") | test_feature["出院诊断病种名称"].str.contains("癫痫") | test_feature["出院诊断病种名称"].str.contains("中风")).astype(int)
    # test_feature["精神疾病"] = (test_feature["出院诊断病种名称"].str.contains("精神") | test_feature["出院诊断病种名称"].str.contains("抑郁") | test_feature["出院诊断病种名称"].str.contains("失眠") | test_feature["出院诊断病种名称"].str.contains("焦虑")).astype(int)
    # test_feature["重症疾病"] = 0
    # for t in SERIOUS_DISEASE:
    #     train_feature["重症疾病"] += train_feature["出院诊断病种名称"].str.contains(t).astype(int)
    #     test_feature["重症疾病"] += test_feature["出院诊断病种名称"].str.contains(t).astype(int)
    # train_feature["重症疾病"] = train_feature["重症疾病"].map(lambda x: 1 if x > 0 else 0)
    # test_feature["重症疾病"] = test_feature["重症疾病"].map(lambda x: 1 if x > 0 else 0)
    # train_feature.to_csv("./data_tmp/train4.csv", index=False)
    # test_feature.to_csv("./data_tmp/test4.csv", index=False)

    # # 造顺序号层面的特征
    # train_feature = pd.read_csv("./data_tmp/train4.csv", encoding="gbk")
    # test_feature = pd.read_csv("./data_tmp/test4.csv", encoding="gbk")
    # train_feature = get_feature(train_feature)
    # test_feature = get_feature(test_feature)
    # train_feature.to_csv("./data_tmp/train5.csv", index=False)
    # test_feature.to_csv("./data_tmp/test5.csv", index=False)

    # # 生成医院额外数据
    # train_feature = pd.read_csv("./data_tmp/train5.csv", encoding="gbk", low_memory=False)
    # train_label = pd.read_csv("./data/df_id_train.csv", header=None)
    # train_label.columns = ["个人编码", "label"]
    # train = train_feature.merge(train_label, on="个人编码", how="left")
    # train = train[["个人编码", "医院编码", "label"]]
    # train_group = train.groupby("医院编码")["label"]
    # for i, (hos_id, data) in enumerate(train_group):
    #     t = pd.Series([hos_id, len(data), data.sum(), len(data) - data.sum(), data.sum() / len(data)], index=["医院编号", "样本总数", "真出现次数", "假出现次数", "频率"])
    #     if i == 0:
    #         res = pd.DataFrame([list(t)], columns=t.index.tolist())
    #     else:
    #         res = res.append(t, ignore_index=True)
    # res[["医院编号", "样本总数", "真出现次数", "假出现次数"]] = res[["医院编号", "样本总数", "真出现次数", "假出现次数"]].astype(int)
    # res.to_csv("hos_info.csv", index=False)

    # # 创建特征
    # train_feature = pd.read_csv("./data_tmp/train5.csv", encoding="gbk", low_memory=False)
    # test_feature = pd.read_csv("./data_tmp/test5.csv", encoding="gbk", low_memory=False)
    # train_user = get_user_feature(train_feature)
    # test_user = get_user_feature(test_feature)
    # train_user.to_csv("./data_tmp/train_user.csv", index=False)
    # test_user.to_csv("./data_tmp/test_user.csv", index=False)

    # # 补充特征
    # # 1. 个人详细情况数据
    # user_details = pd.read_csv("./detail_result.csv", encoding="gbk")
    # train_user = pd.read_csv("./data_extend/train_user.csv", encoding="gbk", low_memory=False)
    # test_user = pd.read_csv("./data_extend/test_user.csv", encoding="gbk", low_memory=False)
    # train_raw = pd.read_csv("./data_extend/train1.csv", encoding="gbk", low_memory=False)[["顺序号", "个人编码"]]
    # test_raw = pd.read_csv("./data_extend/test1.csv", encoding="gbk", low_memory=False)[["顺序号", "个人编码"]]
    # train_feature = train_raw.merge(user_details, on="顺序号", how="left")
    # test_feature = test_raw.merge(user_details, on="顺序号", how="left")
    # def make_feature_detail(input_data):
    #     new_columns_name = ["详细数量", "详细均价", "详细最高价"]
    #     def _max(data, col):
    #         t_max = data[col].max()
    #         t_max.index = [t + "_max" for t in t_max.index.tolist()]
    #         return t_max
    #
    #     def _min(data, col):
    #         t = data[col].min()
    #         t.index = [s + "_min" for s in t.index.tolist()]
    #         return t
    #
    #     def _mean(data, col):
    #         t = data[col].mean()
    #         t.index = [s + "_mean" for s in t.index.tolist()]
    #         return t
    #
    #     group = input_data.groupby("个人编码")
    #     for i, (uid, data) in enumerate(tqdm(group)):
    #         t = pd.Series(uid, index=["uid"])
    #         t = t.append([_max(data, new_columns_name),
    #                       _min(data, new_columns_name),
    #                       _mean(data, new_columns_name)])
    #         if i == 0:
    #             result_df = pd.DataFrame([list(t)], columns=t.index.tolist())
    #         else:
    #             result_df = result_df.append(t, ignore_index=True)
    #     result_df["uid"] = result_df["uid"].astype(np.int64)
    #     return result_df
    #
    # train_user_new = make_feature_detail(train_feature)
    # test_user_new = make_feature_detail(test_feature)
    # train_user = train_user.merge(train_user_new, how="left", on="uid")
    # test_user = test_user.merge(test_user_new, how="left", on="uid")
    # train_user.to_csv("./data_extend/train_user1.csv", index=False)
    # test_user.to_csv("./data_extend/test_user1.csv", index=False)

    # 删除无用特征
    train_user = pd.read_csv("./data_extend/train_user1.csv", encoding="gbk", low_memory=False)
    test_user = pd.read_csv("./data_extend/test_user1.csv", encoding="gbk", low_memory=False)
    for col_name in train_user.columns:
        train_series = train_user[col_name]
        test_series = test_user[col_name]
        if len(train_series.unique()) < 2 or len(test_series.unique()) < 2:
            train_user = train_user.drop(col_name, axis=1)
            test_user = test_user.drop(col_name, axis=1)
    train_user.to_csv("./data_extend/train_user2.csv", index=False)
    test_user.to_csv("./data_extend/test_user2.csv", index=False)