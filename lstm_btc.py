import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def calculate_order_book_features(df):
    # Midpoint-Based Indicators
    df['EMA12_mid'] = df['midpoint'].ewm(span=12, adjust=False).mean()
    df['EMA25_mid'] = df['midpoint'].ewm(span=25, adjust=False).mean()
    df['MACD_mid'] = df['EMA12_mid'] - df['EMA25_mid']
    df['Volatility_mid'] = df['midpoint'].rolling(window=20).std()
    
    # Spread Features
    df['Spread_Avg'] = df['spread'].rolling(window=20).mean()
    df['Spread_Std'] = df['spread'].rolling(window=20).std()
    
    # Order Flow Imbalance
    df['Order_Imbalance'] = (df['buys'] - df['sells']) / (df['buys'] + df['sells'] + 1e-6)  # Avoid division by zero
    
    # Aggregated Liquidity
    df['Total_Bid_Liquidity'] = df[[f'bids_notional_{i}' for i in range(5)]].sum(axis=1)
    df['Total_Ask_Liquidity'] = df[[f'asks_notional_{i}' for i in range(5)]].sum(axis=1)
    df['Liquidity_Imbalance'] = df['Total_Bid_Liquidity'] - df['Total_Ask_Liquidity']
    
    # Weighted Average Distance
    bid_weights = np.arange(1, 6)
    ask_weights = np.arange(1, 6)
    df['Weighted_Bid_Distance'] = (df[[f'bids_distance_{i}' for i in range(5)]] * bid_weights).sum(axis=1) / bid_weights.sum()
    df['Weighted_Ask_Distance'] = (df[[f'asks_distance_{i}' for i in range(5)]] * ask_weights).sum(axis=1) / ask_weights.sum()
    
    df.fillna(0, inplace=True)
    return df

data = pd.read_csv("/kaggle/input/trading-dataset/BTC_1sec.csv").head(150000)
columns_to_keep = [
    "midpoint", "spread", "buys", "sells",
    "bids_distance_0", "bids_distance_1", "bids_distance_2", "bids_distance_3", "bids_distance_4",
    "bids_notional_0", "bids_notional_1", "bids_notional_2", "bids_notional_3", "bids_notional_4",
    "asks_distance_0", "asks_distance_1", "asks_distance_2", "asks_distance_3", "asks_distance_4",
    "asks_notional_0", "asks_notional_1", "asks_notional_2", "asks_notional_3", "asks_notional_4"
]
# Convert system_time to datetime and sort by time
data['system_time'] = pd.to_datetime(data['system_time'])
data.sort_values('system_time', inplace=True)
data.reset_index(drop=True, inplace=True)
data = data[columns_to_keep]
data = calculate_order_book_features(data)

features = data.drop(columns=['midpoint'])
target = data['midpoint']
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_x.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

data.shape

def create_sequences(data, target, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(features_scaled, target_scaled, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

def compute_metrics(y_true, y_pred):
    with torch.no_grad():
        y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
        directional_accuracy = np.mean(np.sign(np.diff(y_true, axis=0)) == np.sign(np.diff(y_pred, axis=0))) * 100
        return rmse, mae, mape, directional_accuracy

input_size = X_train.shape[2]
hidden_size = 32
num_layers = 1
output_size = 1
dropout = 0.75

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

batch_size = 64
train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    batch_metrics = []
    
    for batch in tqdm(train_loader, desc=f"Running Epoch: {epoch+1}"):
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        rmse, mae, mape, direction_acc = compute_metrics(y_batch, y_pred)
        batch_metrics.append((rmse, mae, mape, direction_acc))
    
    avg_metrics = np.mean(batch_metrics, axis=0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}, RMSE: {avg_metrics[0]:.4f}, MAE: {avg_metrics[1]:.4f}, Directional Accuracy: {avg_metrics[3]:.2f}%")

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).cpu().numpy()
    y_test_actual = y_test.cpu().numpy()
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_test_actual = scaler_y.inverse_transform(y_test_actual)
    
    rmse, mae, mape, direction_acc = compute_metrics(torch.tensor(y_test_actual), torch.tensor(y_test_pred))
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, Directional Accuracy: {direction_acc:.2f}%")


plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual', color='blue')
plt.plot(y_test_pred, label='Predicted', color='red', linestyle='dashed')
plt.title("BTC Midpoint Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# save the model
torch.save(model.state_dict(), 'lstm_btc.pth')