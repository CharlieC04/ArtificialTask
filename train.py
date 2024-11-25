import torch.nn as nn
import torch.optim as optim
import pandas as pd


class WineClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, best_params):
        super(WineClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, best_params['num_hidden_units_1'])
        self.fc2 = nn.Linear(best_params['num_hidden_units_1'], best_params['num_hidden_units_2'])
        self.fc3 = nn.Linear(best_params['num_hidden_units_2'], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(best_params['dropout_rate'])

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
def train(params, X_train_combined, y_train, train_loader, device):
    _, class_labels = pd.factorize(y_train)
    # Initialize the model
    final_model = WineClassifier(input_dim=X_train_combined.shape[1], num_classes=len(y_train.unique()), best_params=params)
    final_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=params['learning_rate'])

    epochs = 20

    for epoch in range(epochs):
        final_model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return final_model, class_labels

