# TextPreprocessor
# -lower
# -tokenize
# -remove_stop
# -get_word_frequencies
# -get_word_tfidf
# TextPlot
# -frequency_plot
# -word_cloud


import regex as re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolz import compose
from collections import Counter
from wordcloud import WordCloud


class TextPreprocessor:
    def __init__(self, df):
        self.df = df

    def remove_stopwords(self, tokens, additional_stopwords={""}):
        """
        removes stopwords from a given list of tokens.
        additional_stopwords append to the nltk english corpus.
        """
        stopwords = set(nltk.corpus.stopwords.words("english"))
        stopwords |= additional_stopwords

        return [t for t in tokens if t.lower() not in stopwords]

    def tokenize(self, text):
        """
        Takes a sentence and splits it up into its constituent unicode pieces
        """
        # \p{L} matches all unicode letters
        return re.findall(r"[\w-]*\p{L}[\w-]*", text)

    def prep(self, text_column="text", rm_stopwords=True, save=False):
        """
        Performs case folding, tokenization, and stop word removal.
        Returns a df with an additional tokens column.
        pass save to overwrite self.df as the result
        """
        # define operations
        pipeline = (
            [str.lower, self.tokenize, self.remove_stopwords]
            if rm_stopwords is True
            else [str.lower, self.tokenize]
        )

        def prepare(text):
            # reverses the pipeline and calls it in reverse-order on the text
            return compose(*pipeline[::-1])(text)

        result = self.df.assign(tokens=self.df[text_column].apply(prepare))
        if save:
            self.df = result
            return self
        return result

    def word_counts(self, column="tokens", min_freq=2, save=False):
        """
        Returns a standalone data frame which contains the counts for all the words.
        Text must use .prep() or tokenize() first!
        """
        # create a counter and run through all data
        counter = Counter()
        self.df[column].map(counter.update)

        # transform counter into a DataFrame
        freq_df = (
            pd.DataFrame.from_dict(counter, orient="index", columns=["freq"])
            .query("freq >= @min_freq")
            .sort_values("freq", ascending=False)
        )
        freq_df.index.name = "token"

        if save:
            self.freq_df = freq_df
            return self

        return freq_df

    def attach_tfidf(self, min_df=2, save=False):
        """
        Requires both tokenization and a frequency df!
        WARNING: Not too certain about this. I took a bunch from a book
        but I also twiddled things and didn't do any testing
        """

        def compute_idf(min_df=2):
            def update(doc):
                counter.update(set(doc))

            # count tokens
            counter = Counter()
            self.df["tokens"].map(update)

            # create a df and compute idf
            idf_df = pd.DataFrame.from_dict(
                counter, orient="index", columns=["df"]
            ).query("df >= @min_df")

            idf_df["idf"] = np.log(len(self.df) / idf_df["df"]) + 0.1
            idf_df.index.name = "token"
            return idf_df

        # def update(doc):
        #     tokens = doc if preprocess is None else
        #     preprocess(doc)
        #     counter.update(set(tokens))
        # # count tokens
        # counter = Counter()
        # df[column].map(update)
        # # create DataFrame and compute idf
        # idf_df = pd.DataFrame.from_dict(counter, orient='index',
        #                                 columns=['df'])
        # idf_df = idf_df.query('df >= @min_df')
        # idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
        # idf_df.index.name = 'token'
        # return idf_df

        # def compute_idf(min_df=min_df):
        #     # create data frame and compute idf
        #     self.freq_df["idf"] = (
        #         np.log(len(self.df) / self.freq_df.query("freq > @min_df")["freq"])
        #         + 0.1
        #     )
        #     return self.freq_df

        tf_idf = self.freq_df["freq"] * compute_idf()["idf"]

        if save:
            self.tfidf = tf_idf.fillna(0.001)
            return self
        return tf_idf.fillna(0.001)


un_prep = TextPreprocessor(un[un["year"] == 1970])


# print(un := un_prep.prep("text"))

print(un_prep.prep("text", save=True).word_counts(save=True).attach_tfidf(save=True))

print(un_prep.tfidf)

# print(un_prep.attach_tfidf())

# print(un_prep.freq_df["freq"])

# print(un_prep.df)


class TextPlot(TextPreprocessor):
    def __init__(self, df):
        super().__init__(df)

    def freq_plot(self, top_n=15):
        def make_plot(freq_df):
            plt.close()
            ax = freq_df.head(top_n).plot(kind="barh", width=0.95)
            ax.invert_yaxis()
            ax.set(xlabel="Frequency", ylabel="Token", title="Top Words")
            plt.show()

        try:
            make_plot(self.freq_df)
        except:
            super().prep(save=True).word_counts(save=True)
            make_plot(self.freq_df)

    def wordcloud(self, p_type="freq", title=None, max_words=100, rm_top_n=None):
        def make_wordcloud(counter):
            # instantiate new wordcloud
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="Paired",
                max_font_size=150,
                max_words=max_words,
            )

            # coerce counter to dictionary
            counter = counter.fillna(0).to_dict()

            # filter additional stopwords if necessary
            if rm_top_n is not None:
                # get top n words
                top_n_words = self.freq_df.head(rm_top_n).index
                counter = {
                    token: freq
                    for (token, freq) in counter.items()
                    if token not in top_n_words
                }

            # plot it
            plt.close()
            wc.generate_from_frequencies(counter)
            plt.title(title)
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.show()

        if p_type == "freq":
            try:
                make_wordcloud(self.freq_df["freq"])
            except:
                super().prep(save=True).word_counts(save=True)
                make_wordcloud(self.freq_df["freq"])
        elif p_type == "tf_idf":
            try:
                make_wordcloud(self.tfidf)
            except:
                # this should check if word count exists. if not, gen word count
                # then gen tf idf. instead it just bulldozes ahead
                super().prep(save=True).word_counts(save=True).attach_tfidf(save=True)
                make_wordcloud(self.tfidf)
        else:
            print("Please try type = 'freq' or type = 'tf_idf'")


un_plots = TextPlot(un[un["year"] == 2015])

un_plots.freq_plot(top_n=20)

un_plots.wordcloud(p_type="tf_idf")

# # print(un_plots.freq_df.index)

# # print(un[(un["country"] == "USA")])

# print(un_plots.tfidf)
