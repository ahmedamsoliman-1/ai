import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def visualize_data(self):
        sns.pairplot(self.df)
        plt.show()
