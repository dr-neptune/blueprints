class Overview:
    def __init__(self, df):
        self.df = df

    def summary_stats(self, mem_usage = "deep"):
        """
        Returns a dictionary containing the following summary stats:

        - col names: df.dtype
        - data types + memory consumption: df.info
          - set mem_usage to "" if you don't want to spend more time on "deeper" memory estimates
        - summary: df.describe
        """
        return {
            "col_names": self.df.columns(),
            "data_types": self.df.info(memory_usage = mem_usage),
            "summary": self.df.describe()
        }

un_overview = Overview(un)

print(un_overview.summary_stats())
