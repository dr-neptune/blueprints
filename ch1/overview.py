# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Overview:
    """
    This class instantiates an object that provides an overview of a data frame.
    Example:

    >> overview = Overview(df)
    # get summary statistics
    >> overview.summary_stats()
    ## check for missing values
    >> overview.check_missing()
    ## generate a specific univariate plot
    >> overview.gen_uni_plot("column_name")
    ## generate all univariate plots
    >> overview.gen_all_unis()
    """

    def __init__(self, df):
        self.df = df

    def summary_stats(self, mem_usage="deep", include="O"):
        """
        Returns a dictionary containing the following summary stats:

        - col names: df.dtype
        - data types + memory consumption: df.info
          - set mem_usage to "" if you don't want to spend more time on "deeper" memory estimates
        - summary: df.describe
          - set include to "" if you don't wish to include categorical variables
        """
        column_names = list(self.df.columns)
        # returns a function. Evaluate to get info.
        ## This is because df.info is just a print side effect
        data_types = lambda: self.df.info(memory_usage=mem_usage)
        summary = self.df.describe(include=include).T

        return {
            "col_names": column_names,
            "data_types": data_types,
            "summary": summary,
        }

    def check_missing(self):
        """
        Returns the counts of missing values in the dataframe
        """
        return self.df.isna().sum()

    def gen_uni_plot(self, column_name):
        """
        Generates a univariate density plot for the given column name. Requires a numeric or datetime column
        """
        new_plot = UnivariatePlot(self.df, column_name)
        new_plot.gen_plot()

    def gen_all_unis(self):
        # the [:-1] is because the text field is too large to fix in the axis labels
        return [self.gen_uni_plot(i) for i in self.summary_stats()["col_names"][:-1]]


# un_overview = Overview(un)
# un_overview.gen_all_unis()


class UnivariatePlot:
    sns.set(palette="colorblind")

    def __init__(self, df, column_name, keep_null=False):
        self.column_name = column_name
        # if you wish to keep the null values, pass True to keep_null
        if keep_null:
            self.df = df[column_name].to_frame()
        else:
            self.df = df[column_name].dropna().to_frame()

    def gen_dist_plot(self):
        """
        Generates a univariate density plot for the given column name. Requires a numeric or datetime column
        """
        plt.close()
        # plot a single distribution plot
        sns.displot(data=self.df, kind="kde")
        sns.rugplot(data=self.df)
        plt.title(self.column_name.title())
        plt.show()

    def gen_count_plot(self, top_n=10):
        """
        Generates a count plot for the given column name.
        Returns @top_n values ordered by highest cardinality
        """

        plt.close()
        sns.countplot(
            y=self.column_name,
            data=self.df,
            order=self.df[self.column_name].value_counts().iloc[:top_n].index,
        )
        plt.title(self.column_name.title())
        plt.show()

    def gen_plot(self):
        if self.df[self.column_name].dtype == "object":
            self.gen_count_plot()
        elif self.df[self.column_name].dtype in ["int64", "datetime", "float"]:
            self.gen_dist_plot()
        else:
            raise ValueError("Column type not in [object, int64, datetime, float]")


# un_len = UnivariatePlot(un, "text")
# un_position = UnivariatePlot(df=un, column_name="country")
# un_position.gen_plot()
# un_len.gen_plot()
