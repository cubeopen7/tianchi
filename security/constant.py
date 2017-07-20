# -*- coding: utf-8 -*-

__all__ = ["FEATURE", "USELESS_COLUMNS"]

from collections import OrderedDict

FEATURE = OrderedDict()
FEATURE["顺序号"] = "id"
FEATURE["个人编码"] = "uid"
FEATURE["医院编码"] = "hid"

FEATURE["药品费发生金额"] = "drug_cash"
FEATURE["贵重药品发生金额"] = "valuable_drug_cash"
FEATURE["中成药费发生金额"] = "chinese_patent_drug_cash"
FEATURE["中草药费发生金额"] = "chinese_herb_drug_cash"
FEATURE["药品费自费金额"] = "drug_self_pay"
FEATURE["药品费拒付金额"] = "drug_refuse_pay"
FEATURE["药品费申报金额"] = "drug_declare_pay"

FEATURE["检查费发生金额"] = "examine_cash"
FEATURE["贵重检查费金额"] = "valuable_examine_cash"
FEATURE["检查费自费金额"] = "examine_self_pay"
FEATURE["检查费拒付金额"] = "examine_refuse_pay"
FEATURE["检查费申报金额"] = "examine_declare_pay"

FEATURE["治疗费发生金额"] = "treat_cash"
FEATURE["治疗费自费金额"] = "treat_self_pay"
FEATURE["治疗费拒付金额"] = "treat_refuse_pay"
FEATURE["治疗费申报金额"] = "treat_declare_pay"

FEATURE["手术费发生金额"] = "operation_cash"
FEATURE["手术费自费金额"] = "operation_self_pay"
FEATURE["手术费拒付金额"] = "operation_refuse_pay"
FEATURE["手术费申报金额"] = "operation_declare_pay"

FEATURE["床位费发生金额"] = "bed_cash"
FEATURE["床位费拒付金额"] = "bed_refuse_pay"
FEATURE["床位费申报金额"] = "bed_declare_pay"

FEATURE["医用材料发生金额"] = "medical_material_cash"
FEATURE["医用材料费自费金额"] = "medical_material_self_pay"
FEATURE["医用材料费拒付金额"] = "medical_material_refuse_pay"

FEATURE["高价材料发生金额"] = "valuable_material_cash"

FEATURE["输全血申报金额"] = "transfusion_declare_pay"
FEATURE["输全血按比例自负金额"] = "transfusion_self_pay"
FEATURE["成分输血自费金额"] = "component_transfusion_self_pay"
FEATURE["成分输血拒付金额"] = "component_transfusion_refuse_pay"
FEATURE["成分输血申报金额"] = "component_transfusion_declare_pay"

FEATURE["其它发生金额"] = "others_cash"
FEATURE["其它拒付金额"] = "others_refuse_pay"
FEATURE["其它申报金额"] = "others_declare_pay"

FEATURE["一次性医用材料自费金额"] = "disposable_medical_material_self_pay"
FEATURE["一次性医用材料拒付金额"] = "disposable_medical_material_refuse_pay"
FEATURE["一次性医用材料申报金额"] = "disposable_medical_material_declare_pay"

FEATURE["起付线标准金额"] = "deductible_cash"
FEATURE["起付标准以上自负比例金额"] = "over_deductible_self_pay"
FEATURE["医疗救助个人按比例负担金额"] = "medical_help_self_pay"
FEATURE["最高限额以上金额"] = "over_limit_cash"

FEATURE["统筹拒付金额"] = "total_refuse_pay"

FEATURE["基本医疗保险统筹基金支付金额"] = "fund_pay"
FEATURE["基本医疗保险个人账户支付金额"] = "individual_pay"
FEATURE["残疾军人医疗补助基金支付金额"] = "soldier_pay"
FEATURE["农民工医疗救助计算金额"] = "farmer_assist_cash"
FEATURE["公务员医疗补助基金支付金额"] = "civil_assist_cash"
FEATURE["城乡救助补助金额"] = "rural_assist_cash"
FEATURE["城乡优抚补助金额"] = "rural_comfort_cash"
FEATURE["民政救助补助金额"] = "gov_assist_cash"
FEATURE["非典补助补助金额"] = "sars_assist_cash"
FEATURE["可用账户报销金额"] = "account_cash"
FEATURE["非账户支付金额"] = "account_cash"

FEATURE["双笔退费标识"] = "return_flag"

FEATURE["出院诊断病种名称"] = "disease"

FEATURE["本次审批金额"] = "current_examine_cash"
FEATURE["补助审批金额"] = "assist_examine_cash"

FEATURE["医疗救助医院申请"] = "apply_hospital"

FEATURE["家床起付线剩余"] = "bed_residual"

FEATURE["交易时间"] = "trade_date"
FEATURE["住院开始时间"] = "start_date"
FEATURE["住院终止时间"] = "end_date"
FEATURE["申报受理时间"] = "accept_date"
FEATURE["住院天数"] = "hospital_days"
FEATURE["操作时间"] = "operate_date"

USELESS_COLUMNS = ["药品费拒付金额",  # 0.0
                   "检查费拒付金额",  # 0.0
                   "治疗费拒付金额",  # 0.0
                   "手术费拒付金额",  # 0.0
                   "床位费拒付金额",  # 0.0
                   "医用材料费拒付金额",  # 0.0
                   "输全血申报金额",  # 0.0
                   "成分输血自费金额",  # 0.0
                   "成分输血拒付金额",  # 0.0
                   "其它拒付金额",  # 0.0
                   "一次性医用材料自费金额",  # [0.0, nan]
                   "一次性医用材料拒付金额",  # nan
                   "输全血按比例自负金额",  # 0.0
                   "统筹拒付金额",  # 0.0
                   "农民工医疗救助计算金额",  # nan
                   "双笔退费标识",  # nan
                   "住院天数",  # 0.0
                   "非典补助补助金额",  # [0.0, nan]
                   "家床起付线剩余",  # [0.0, nan]
                   ]