import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


un = pd.read_csv("data/un-debates.csv")

print(un)

# print column names
print(un.columns)

# print column data types
print(un.dtypes)

# data types plus memory consumption
print(un.info())

# summary statistics
print(un.describe())

# summary statistics for all columns (including categorical)
print(un.describe())

# add text length column and describe it
print(un.assign(length=un["text"].str.len()).describe().T)
print(un.assign(length=un["text"].str.len()).describe(include="O").T)

# check number of unique values for categorical predictors
print(un[["country", "speaker"]].describe(include="O").T)

# check for missing values
print(un.isna().sum())

# side by side seaborn plots
where = un["country"].isin(["USA", "FRA", "GBR", "CHN", "RUS"])

sns.catplot(un[where], x="country", y="length", kind="box")
sns.catplot(un[where], x="country", y="length", kind="violin")
plot.show()

# time series viz
print(un)
plt.close("all")


plt.subplot(1, 2, 1)
p1 = un.groupby("year").size().plot(title="Number of Countries")
plt.subplot(1, 2, 2)
p2 = (
    un.groupby("year")
    .agg({"length": "mean"})
    .plot(title="Avg Speech Length", ylim=(0, 30000))
)
plt.show()
