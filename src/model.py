from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

class Model:
    def __init__(self, dataframe, target_column):
        self.X = dataframe.drop(columns=[target_column])
        self.y = dataframe[target_column]
        self.model = RandomForestClassifier()

    def train_and_evaluate(self):
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Training the model
        self.model.fit(X_train, y_train)
        
        # Making predictions on the test set
        y_pred = self.model.predict(X_test)
        
        # Evaluating the model
        report = classification_report(y_test, y_pred)
        return report

    def feature_importance(self):
        # Getting feature importances
        importances = self.model.feature_importances_
        return pd.DataFrame({"Feature": self.X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
