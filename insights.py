import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

un = pd.read_csv("data/un-debates.csv")

print(un)

# print column names
print(un.columns.tolist())

# print column data types
print(un.dtypes)

# data types plus memory consumption
print(un.info())

# summary statistics
print(un.describe())

# summary statistics for all columns (including categorical)
print(un.describe())

# add text length column and describe it
un = un.assign(length=un["text"].str.len())

print(un.assign(length=un["text"].str.len()).describe().T)
print(un.assign(length=un["text"].str.len()).describe(include="O").T)

# check number of unique values for categorical predictors
print(un[["country", "speaker"]].describe(include="O").T)

# check how many NA values we have
print(un.isna().sum())

# fill in the NA with "unknown"
# warning: this is a mutable operation
print(un["speaker"].fillna("unknown", inplace=True))
print(un.isna().sum())

# check specific values and their counts
print(un[un["speaker"].str.contains("Bush")]["speaker"].value_counts())

# plot a box and whisker plot and a histogram side by side
plt.subplot(1, 2, 1)
un["length"].plot(kind="box", vert=False)
plt.subplot(1, 2, 2)
un["length"].plot(kind="hist", bins=30)
plt.title("Speech Length (Characters)")
plt.show()


def gen_dist_plot(column_name):
    plt.close()
    # plot a single distribution plot
    sns.displot(un[column_name], kind="kde")
    sns.rugplot(un[column_name])
    plt.title(column_name.title())
    plt.show()


gen_dist_plot("length")

where = un["country"].isin(["USA", "FRA", "GBR", "CHN", "RUS"])

print(un[where])

sns.catplot(data=un[where], x="country", y="length", kind="box")
sns.catplot(data=un[where], x="country", y="length", kind="violin")
plt.show()

plt.subplot(1, 2, 1)
un.groupby("year").size().plot(title="Number of Countries")
plt.subplot(1, 2, 2)
un.groupby("year").agg({"length": "mean"}).plot(
    title="Avg Speech Length", ylim=(0, 30000)
)
plt.show()

import regex as re


def tokenize(text):
    # \p{L} matches all unicode letters
    return re.findall(r"[\w-]*\p{L}[\w-]*", text)


text = "Let's defeat SARS-CoV-2 together in 2020!"

print("|".join(tokens := tokenize(text)))

stopwords = set(nltk.corpus.stopwords.words("english"))


def remove_stopwords(tokens):
    return [t for t in tokens if t.lower() not in stopwords]


# adding additional stopwords
include_stopwords = {"dear", "regards", "must", "would", "also"}
exclude_stopwords = {"against"}

stopwords |= include_stopwords
stopwords -= exclude_stopwords

from toolz import compose

pipeline = [str.lower, tokenize, remove_stopwords]

# lol
def prepare(text, pipeline):
    # reverses the pipeline and calls it in reverse-order on the text
    return compose(*pipeline[::-1])(text)


# applying prepare to a dataframe
un = un.assign(
    tokens=un["text"].apply(prepare, pipeline=pipeline),
    num_tokens=un["tokens"].map(len),
)

print(un[["text", "tokens", "num_tokens"]])

from pandarallel import pandarallel
import math

pandarallel.initialize(progress_bar=True, verbose=2)

df_size = int(5e6)
df = pd.DataFrame({"a": np.random.randint(1, 8, df_size), "b": np.random.rand(df_size)})

# df.apply
res_parallel = df.parallel_apply(
    lambda x: math.sin(x.a ** 2) + math.sin(x.b ** 2), axis=1
)
print(res_parallel)

# series.map
df_size = int(1e7)
df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=np.random.rand(df_size)))


def func(x):
    return math.sin(x ** 2) - math.cos(x ** 2)


df = df.assign(results=df["a"].parallel_map(func))

print(df)

from collections import Counter

tokens = tokenize("She likes my cats and my cats like my sofa")

print(counter := Counter(tokens))

# update the counter
more_tokens = tokenize("She likes dogs and cats.")
counter.update(more_tokens)
print(counter)

# get the counts of all the tokens in the df
counter = Counter()

un["tokens"].map(counter.update)

print(counter.most_common(5))

# make the counter into a dataframe
def count_words(df, column="tokens", preprocess=None, min_freq=2):
    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)

    # create a counter and run through all data
    counter = Counter()
    df[column].map(counter.update)

    # transform counter into a DataFrame
    freq_df = pd.DataFrame.from_dict(counter, orient="index", columns=["freq"]).query(
        "freq >= @min_freq"
    )
    freq_df.index.name = "token"

    return freq_df.sort_values("freq", ascending=False)
