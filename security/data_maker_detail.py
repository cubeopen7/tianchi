# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import f_classif

detail = pd.read_csv("./data/fee_detail.csv")
detail["费用发生时间"] = detail["费用发生时间"].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
detail_group = detail.groupby("顺序号")

for i, (id, data) in enumerate(tqdm(detail_group)):
    t = [id]
    # 1. 详细数量
    t.append(len(data))
    # 2. 详细均价
    price = data["单价"]
    amount = data["数量"]
    total = price * amount
    t2 = total.mean()
    t.append(t2)
    # 3. 详细最高价
    t3 = total.max()
    t.append(t3)
    # 4. 详细间隔日期
    t4 = (data["费用发生时间"].max() - data["费用发生时间"].min()).days
    t.append(t4)
    # 汇总
    if i == 0:
        detail_result = pd.DataFrame([t], columns=["顺序号", "详细数量", "详细均价", "详细最高价", "详细间隔日期"])
    else:
        detail_result = detail_result.append(pd.Series(t, index=["顺序号", "详细数量", "详细均价", "详细最高价", "详细间隔日期"]), ignore_index=True)
detail_result.to_csv("detail_result.csv", index=False)