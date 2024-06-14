import seaborn as sns
import matplotlib.pyplot as plt

class Visualize:
    def __init__(self, feature_importances):
        self.feature_importances = feature_importances
    
    def plot_feature_importance(self):
        sns.barplot(x="Importance", y="Feature", data=self.feature_importances)
        plt.title("Feature Importance")
        plt.show()
