
# 导入库
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# 导入数据集
data = pd.read_csv("lung.csv", index_col = 0)
data.head()

data.shape
data.dtypes

data = data[['time', 'status', 'age', 'sex', 'ph.ecog', 'ph.karno','pat.karno', 'meal.cal', 'wt.loss']]

data["status"] = data["status"] - 1
data["sex"] = data["sex"] - 1
data.head()

data.isnull().sum()

data["ph.karno"] = data["ph.karno"].fillna(data["ph.karno"].mean())
data["pat.karno"] = data["pat.karno"].fillna(data["pat.karno"].mean())
data["meal.cal"] = data["meal.cal"].fillna(data["meal.cal"].mean())
data["wt.loss"] = data["wt.loss"].fillna(data["wt.loss"].mean())

data.dropna(inplace=True)
data.isnull().sum()

data["ph.ecog"] = data["ph.ecog"].astype("int64")

data['ph.ecog'].value_counts()

data = data[data["ph.ecog"] != 3]
data.shape


dummies_ecog = pd.get_dummies(data["ph.ecog"], prefix='ecog')
dummies_ecog.head(4)

dummies_ecog = dummies_ecog[["ecog_1", "ecog_2"]]

data = pd.concat([data, dummies_ecog], axis=1)

data = data.drop("ph.ecog", axis=1)

cph = CoxPHFitter()
cph.fit(data, duration_col='time', event_col='status')

cph.print_summary()

plt.subplots(figsize=(10, 6))
cph.plot()

import numpy as np

# 获取基线生存函数
survival_function = cph.baseline_survival_

# 找到中位生存时间，即生存函数值为 0.5 的时间点
median_survival_time = survival_function[survival_function['baseline survival'] <= 0.5].index[0]
print("Median survival time:", median_survival_time)

from lifelines.utils import median_survival_times

# 使用bootstrap方法估计中位生存时间的置信区间
bootstrapped_median_times = []
for _ in range(100):  # 例如，重复100次抽样
    # 从原始数据中抽取样本，替换=True表示放回抽样
    sample = data.sample(n=len(data), replace=True)
    cph.fit(sample, duration_col='time', event_col='status', show_progress=False)
    sample_survival_function = cph.baseline_survival_
    sample_median_time = sample_survival_function[sample_survival_function['baseline survival'] <= 0.5].index[0]
    bootstrapped_median_times.append(sample_median_time)

# 计算置信区间
lower = np.percentile(bootstrapped_median_times, 2.5)
upper = np.percentile(bootstrapped_median_times, 97.5)
print("95% Confidence Interval for the Median Survival Time: ({}, {})".format(lower, upper))

cph.plot_partial_effects_on_outcome(covariates = 'age', values = [50, 60, 70, 80], cmap = 'coolwarm')

cph.plot_partial_effects_on_outcome(covariates = 'sex', values = [0,1], cmap = 'coolwarm')

cph.check_assumptions(data, p_value_threshold = 0.05)

from lifelines.statistics import proportional_hazard_test

results = proportional_hazard_test(cph, data, time_transform='rank')
results.print_summary(decimals=3, model="untransformed variables")