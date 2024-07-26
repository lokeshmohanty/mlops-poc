import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utils import plot_decision_boundary
from clearml import Task

class Adaboost:
    def __init__(self, project_name, task_name, data):
        task = Task.init(
            project_name=project_name,
            task_name=task_name
        )
        self.X_train, self.X_test, self.y_train, self.y_test = data


    def train(self): 
        self.model = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1), n_estimators=30,
            learning_rate=0.5, random_state=42)
        self.model.fit(self.X_train, self.y_train)


    def eval(self):
        y_pred = model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        plot_decision_boundary(model, self.X_train, self.y_train)
        task.logger.report_scalar("Performance", "Accuracy", value=accuracy, iteration=0)
        task.connect({"n_estimators": 100, "learning_rate": 0.5, "random_state": 42})
        task.upload_artifact("model", model)
        print(f"Accuracy: {accuracy}")
