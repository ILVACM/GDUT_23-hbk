
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

data.isnull().sum()

# 填补缺失值
data["ph.karno"].fillna(data["ph.karno"].mean(), inplace=True)
data["pat.karno"].fillna(data["pat.karno"].mean(), inplace=True)
data["meal.cal"].fillna(data["meal.cal"].mean(), inplace=True)
data["wt.loss"].fillna(data["wt.loss"].mean(), inplace=True)

data.dropna(inplace=True)
data.isnull().sum()

data["ph.ecog"] = data["ph.ecog"].astype("int64")

E = data['status']

T = data["time"]
plt.hist(T, bins = 50)
plt.show()

kmf = KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E)
kmf.plot_survival_function()
plt.title('Survival function')

kmf.median_survival_time_

ax = plt.subplot(111)
m = (data["sex"] == 0)
kmf.fit(durations=T[m], event_observed=E[m], label="Male")
kmf.plot_survival_function(ax=ax)
kmf.fit(T[~m], event_observed=E[~m], label="Female")
kmf.plot_survival_function(ax=ax, at_risk_counts=True)
plt.title("Survival of different gender groups")

ecog_types = data.sort_values(by=['ph.ecog'])['ph.ecog'].unique()
for i, ecog_type in enumerate(ecog_types):
    ax = plt.subplot(2, 2, i + 1)
    ix = data['ph.ecog'] == ecog_type
    kmf.fit(T[ix], E[ix], label=str(ecog_type))
    kmf.plot_survival_function(ax=ax)
    plt.title(f'ECOG {ecog_type}')
plt.tight_layout()