# %%
# # pip install lifelines --user
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("lung.csv", index_col = 0)
data.head()

# %%
data.shape

# %%
data.dtypes

# %%
data = data[['t1me', 'status', 'age', 'sex', 'myc', 'smok1ng','gene_alterat1on_status', 'patholog1cal_stage', 'pstage_1or2']]
data["status"] = data["status"] - 1
data["sex"] = data["sex"] - 1
data.head()

# %%
data.isnull().sum()

# %%
data.columns

# %%
# 直接替换整列
data["smok1ng"] = data["smok1ng"].fillna(data["smok1ng"].mean())
data["gene_alterat1on_status"] = data["gene_alterat1on_status"].fillna(data["gene_alterat1on_status"].mean())
data["patholog1cal_stage"] = data["patholog1cal_stage"].fillna(data["patholog1cal_stage"].mean())
data["pstage_1or2"] = data["pstage_1or2"].fillna(data["pstage_1or2"].mean())
data.dropna(inplace=True)
data["myc"] = data["myc"].astype("int64")

# %%
data.isnull().sum()

# %%
data.shape

# %%
T = data["t1me"]
E = data["status"]
plt.hist(T, bins = 50)
plt.show()

# %%
#Fitting a non-parametric model [Kaplan Meier Curve]
kmf = KaplanMeierFitter()
kmf.fit(durations = T, event_observed = E)
kmf.plot_survival_function()

# %%
kmf.survival_function_.plot()
plt.title('Survival function')

# %%
kmf.plot_cumulative_density()

# %%
kmf.median_survival_time_

# %%
from lifelines.utils import median_survival_times

median_ = kmf.median_survival_time_
median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
print(median_)
print(median_confidence_interval_)

# %%
ax = plt.subplot(111)

m = (data["sex"] == 0)

kmf.fit(durations = T[m], event_observed = E[m], label = "Male")
kmf.plot_survival_function(ax = ax)

kmf.fit(T[~m], event_observed = E[~m], label = "Female")
kmf.plot_survival_function(ax = ax, at_risk_counts = True)

plt.title("Survival of different gender group")

# %% [markdown]
# 

# %%
ecog_types = data.sort_values(by = ['myc'])["myc"].unique()

for i, ecog_types in enumerate(ecog_types):
    ax = plt.subplot(2, 2, i + 1)
    ix = data['myc'] == ecog_types
    kmf.fit(T[ix], E[ix], label = ecog_types)
    kmf.plot_survival_function(ax = ax, legend = False)
    plt.title(ecog_types)
    plt.xlim(0, 1200)

plt.tight_layout()

# %%



