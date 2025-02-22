import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = "/kaggle/input/trading-dataset/Lobster_data/GOOG_orderbook.csv"
df = pd.read_csv(file_path, header=None)


df = df / 10000.0  
num_levels = 3
features = []
for i in range(num_levels):
    ask_price = df.iloc[:, 4 * i]   # Ask Price i
    ask_size = df.iloc[:, 4 * i + 1]   # Ask Volume i
    bid_price = df.iloc[:, 4 * i + 2]   # Bid Price i
    bid_size = df.iloc[:, 4 * i + 3]   # Bid Volume i
    features.extend([ask_price, ask_size, bid_price, bid_size])


X = pd.concat(features, axis=1).values  # Shape: (num_samples, num_features)
y = df.iloc[:, 2]  # Predicting best bid price (can modify to predict mid-price)


X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y.values, dtype=torch.float32, device=device).view(-1, 1)


# Create sequences for RNN (e.g., lookback of 10 steps)
sequence_length = 10

def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length - 1):
        X_seq.append(X[i : i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return torch.stack(X_seq), torch.stack(y_seq)

# Create sequences BEFORE splitting
X_seq, y_seq = create_sequences(X_tensor, y_tensor, sequence_length)

# Now split into training and testing sets
from sklearn.model_selection import train_test_split
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False  # shuffle=False prevents time-mixing
)


# Normalize only on training data
X_train_mean, X_train_std = X_train_seq.mean(dim=0), X_train_seq.std(dim=0)
X_train_seq = (X_train_seq - X_train_mean) / (X_train_std + 1e-8)
X_test_seq = (X_test_seq - X_train_mean) / (X_train_std + 1e-8)  # Use training mean/std!

batch_size = 32
train_dataset = TensorDataset(X_train_seq, y_train_seq)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_seq, y_test_seq)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LOB_RNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.5):
        super(LOB_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

input_size = X_train_seq.shape[2]
model = LOB_RNN(input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    running_loss = 0.0
    total_samples = 0
    all_predictions_train, all_targets_train = [], []
    for batch_X, batch_y in pbar:
        batch_size_current = batch_X.size(0)
        total_samples += batch_size_current

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size_current

        current_loss = running_loss / total_samples

        # Calculate metrics
        with torch.no_grad():
            all_predictions = predictions.cpu().numpy()
            all_targets = batch_y.cpu().numpy()

            mse_loss = np.mean((all_predictions - all_targets) ** 2)
            rmse_loss = np.sqrt(mse_loss)
            mae_loss = np.mean(np.abs(all_predictions - all_targets))
            r2_score = 1 - (np.sum((all_targets - all_predictions) ** 2) /
                            np.sum((all_targets - np.mean(all_targets)) ** 2))
            percentage_loss = rmse_loss / np.mean(all_targets) * 100

        all_predictions_train.append(all_predictions)
        all_targets_train.append(all_targets)

        pbar.set_postfix({
            "Loss": f"{current_loss:.5f}",
            "RMSE": f"{rmse_loss:.5f}",
            "MAE": f"{mae_loss:.5f}",
            "R²": f"{r2_score:.5f}",
            "Percentage Error": f"{percentage_loss:.4f}%"
        })

    print(f"Epoch {epoch+1}, Loss: {current_loss:.5f}, RMSE: {rmse_loss:.5f}, "
          f"MAE: {mae_loss:.5f}, R²: {r2_score:.5f}, Percentage Error: {percentage_loss:.4f}%")
    

all_predictions_train = np.concatenate(all_predictions_train, axis=0)
all_targets_train = np.concatenate(all_targets_train, axis=0)

plt.figure(figsize=(10,5))
plt.plot(all_targets_train, label="Actual Prices")
plt.plot(all_predictions_train, label="Predicted Prices", linestyle="dashed")
plt.legend()
plt.title("LOBSTER Model Predictions vs Actual Prices")
plt.show()

model.eval()
with torch.no_grad():
    all_predictions_test = []
    all_targets_test = []
    for batch_X, batch_y in test_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X)
        all_predictions_test.append(predictions.cpu().numpy())
        all_targets_test.append(batch_y.cpu().numpy())

    all_predictions_test = np.concatenate(all_predictions_test, axis=0)
    all_targets_test = np.concatenate(all_targets_test, axis=0)

    mse_loss = np.mean((all_predictions_test - all_targets_test) ** 2)
    rmse_loss = np.sqrt(mse_loss)
    mae_loss = np.mean(np.abs(all_predictions_test - all_targets_test))
    r2_score = 1 - (np.sum((all_targets_test - all_predictions_test) ** 2) /
                    np.sum((all_targets_test - np.mean(all_targets_test)) ** 2))
    percentage_loss = rmse_loss / np.mean(y_test_seq.cpu().numpy()) * 100

    print(f"Test set results - RMSE: {rmse_loss:.5f}, MAE: {mae_loss:.5f}, "
          f"R²: {r2_score:.5f}, Percentage Error: {percentage_loss:.4f}%")

plt.figure(figsize=(10,5))
plt.plot(all_targets_test, label="Actual Prices")
plt.plot(all_predictions_test, label="Predicted Prices", linestyle="dashed")
plt.legend()
plt.title("LOBSTER Model Predictions vs Actual Prices")
plt.show()
