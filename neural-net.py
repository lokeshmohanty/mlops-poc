from clearml import Task
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_data, plot_decision_boundary

# Initialize the task
task = Task.init(project_name="Synthetic Classification", task_name="Neural Classifier")

# Get the data
X_train, X_test, y_train, y_test = get_data()

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def predict(self, x):
        x = self.forward(torch.FloatTensor(x))
        return x.detach().numpy()

# Initialize the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100 #0
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        task.logger.report_scalar("Performance",
                                  "Loss",
                                  value=loss.item(),
                                  iteration=i)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    predicted = (test_outputs > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    
    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Test Accuracy: {accuracy.item():.4f}')

    task.logger.report_scalar("Performance",
                              "Accuracy",
                              value=accuracy.item(),
                              iteration=i)

plot_decision_boundary(model,
                       X_train.detach().numpy(),
                       y_train.detach().numpy().reshape(-1))
# Log hyperparameters
task.connect({"num_epochs": 1000, "learning_rate": 0.01})

# Save the model
task.upload_artifact("model", model)
