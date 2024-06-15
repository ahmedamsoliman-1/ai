import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, dataframe):
        self.df = dataframe

    def visualize_data(self):
        # Visualization: Outcome distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='result')
        plt.title('Game Outcome Distribution')
        plt.show()

        # Visualization: Game duration distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='game_duration', bins=30)
        plt.title('Game Duration Distribution')
        plt.show()

        # Visualization: Captured pieces count
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='captured_piece_byblack_count', color='black', alpha=0.5, label='Black')
        sns.histplot(data=self.df, x='captured_piece_bywhite_count', color='white', alpha=0.5, label='White')
        plt.legend()
        plt.title('Captured Pieces Distribution')
        plt.show()
