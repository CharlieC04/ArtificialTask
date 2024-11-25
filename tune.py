import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

# Define the objective function for hyperparameter optimization
def objective(trial, X_train_combined, y_train, device, train_loader, test_loader):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    num_hidden_units_1 = trial.suggest_int('num_hidden_units_1', 64, 256)
    num_hidden_units_2 = trial.suggest_int('num_hidden_units_2', 32, 128)
    
    # Define the model with trial parameters
    class TunableWineClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(TunableWineClassifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, num_hidden_units_1)
            self.fc2 = nn.Linear(num_hidden_units_1, num_hidden_units_2)
            self.fc3 = nn.Linear(num_hidden_units_2, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = TunableWineClassifier(input_dim=X_train_combined.shape[1], num_classes=len(y_train.unique()))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = 10  # Reduce epochs for faster tuning
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    return accuracy

def tune(X_train_combined, y_train, train_loader, test_loader, device):

    objective_with_params = partial(
        objective,
        train_loader=train_loader,
        test_loader=test_loader,
        X_train_combined=X_train_combined,
        y_train=y_train,
        device=device
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_with_params, n_trials=20)

    return study.best_params
