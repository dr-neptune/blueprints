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
un = un.assign(length = un["text"].str.len())

print(un.assign(length=un["text"].str.len()).describe().T)
print(un.assign(length=un["text"].str.len()).describe(include="O").T)

# check number of unique values for categorical predictors
print(un[["country", "speaker"]].describe(include="O").T)

# check how many NA values we have
print(un.isna().sum())

# fill in the NA with "unknown"
# warning: this is a mutable operation
print(un["speaker"].fillna("unknown", inplace = True))
print(un.isna().sum())

# check specific values and their counts
print(un[un["speaker"].str.contains("Bush")]["speaker"].value_counts())

# plot a box and whisker plot and a histogram side by side
plt.subplot(1, 2, 1)
un["length"].plot(kind = "box", vert = False)
plt.subplot(1, 2, 2)
un["length"].plot(kind = "hist", bins = 30)
plt.title("Speech Length (Characters)")
plt.show()

where = un["country"].isin(["USA", "FRA", "GBR", "CHN", "RUS"])

print(un[where])

sns.catplot(data = un[where], x="country", y="length", kind="box")
sns.catplot(data = un[where], x="country", y="length", kind="violin")
plt.show()

plt.subplot(1, 2, 1)
un.groupby("year").size().plot(title = "Number of Countries")
plt.subplot(1, 2, 2)
un.groupby("year").agg({"length": "mean"}).plot(title = "Avg Speech Length", ylim = (0, 30000))
plt.show()
