from clearml import Task
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils import plot_decision_boundary

# Initialize the task
task = Task.init(project_name="Synthetic Classification", task_name="Random Forest Classifier")

# Generate some dummy data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

plot_decision_boundary(model, X_train, y_train)

# Log the results
task.logger.report_scalar("Performance", "Accuracy", value=accuracy, iteration=0)

# Log hyperparameters
task.connect({"n_estimators": 100, "random_state": 42})

print(f"Accuracy: {accuracy}")

# Save the model
task.upload_artifact("model", model)

print("Task completed. Check the ClearML web interface for results.")
