from clearml import Task
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from utils import get_data, plot_decision_boundary

# Initialize the task
task = Task.init(project_name="Synthetic Classification", task_name="AdaBoost Classifier")

# Get the data
X_train, X_test, y_train, y_test = get_data()

# Create and train the model
model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

plot_decision_boundary(model, X_train, y_train)

# Log the results
task.logger.report_scalar("Performance", "Accuracy", value=accuracy, iteration=0)

# Log hyperparameters
task.connect({"n_estimators": 100, "learning_rate": 0.5, "random_state": 42})

print(f"Accuracy: {accuracy}")

# Save the model
task.upload_artifact("model", model)

print("Task completed. Check the ClearML web interface for results.")
