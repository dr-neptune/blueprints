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
