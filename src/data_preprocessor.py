class DataPreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def preprocess(self):
        # Example: dropping missing values
        self.df.dropna(inplace=True)
        return self.df
