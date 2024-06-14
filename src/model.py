from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class Model:
    def __init__(self, dataframe, target_column):
        self.X = dataframe.drop(target_column, axis=1)
        self.y = dataframe[target_column]
        self.model = RandomForestClassifier()
    
    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def feature_importance(self):
        importances = self.model.feature_importances_
        return pd.DataFrame({"Feature": self.X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
