import pandas as pd
import numpy as np
import nltk
from yaml import compose

stopwords = set(nltk.corpus.stopwords.words("english"))

from toolz import compose

pipeline = [str.lower, tokenize, remove_stopwords]

print()


def prepare(text, pipeline):
    # reverses the pipeline and calls it in reverse-order on the text
    return compose(*pipeline[::-1])(text)


print(prepare(text, pipeline))

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
