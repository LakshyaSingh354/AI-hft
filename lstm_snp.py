import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {device}")
def calculate_technical_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA25'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA25']
    df['Bollinger_Up'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
    df['Bollinger_Down'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
    df.fillna(0, inplace=True)
    return df
df = pd.read_csv('/kaggle/input/snp-500-intraday-data/dataset.csv')
df.dropna(inplace=True)
df = df.iloc[:,1:5].set_axis(['Open', "High", 'Low', "Close"], axis=1)
df = calculate_technical_indicators(df)
scaler = MinMaxScaler()
data = scaler.fit_transform(df)
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 3])  # Predict 'Close' price
    return np.array(sequences), np.array(targets)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for seqs, targets in tqdm(train_loader, desc=f"Training Epoch: {epoch}"):
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(seqs)
            loss = criterion(output.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate on test data
        model.eval()
        test_loss, test_rmse = 0, 0
        with torch.no_grad():
            for seqs, targets in test_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                output = model(seqs)
                test_loss += criterion(output.squeeze(), targets).item()
                test_rmse += mean_squared_error(targets.cpu().numpy(), output.cpu().numpy(), squared=False)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Test RMSE: {test_rmse / len(test_loader):.6f}")

seq_length = 10
X, y = create_sequences(data, seq_length)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
train_dataset, test_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = LSTMModel(input_size=9, hidden_size=50, num_layers=2, output_size=1).to(device)
train_model(model, train_loader, test_loader, epochs=10, device=device)
def visualize_predictions(model, test_loader, device='cpu'):
    model.eval()
    actuals, predictions = [], []
    with torch.no_grad():
        for seqs, targets in test_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            output = model(seqs)
            actuals.extend(targets.cpu().numpy())
            predictions.extend(output.cpu().numpy())
    
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual Prices", color='blue')
    plt.plot(predictions, label="Predicted Prices", color='red')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title("S&P 500 Intraday Price Prediction")
    plt.show()

visualize_predictions(model, test_loader, device=device)

# Save the model
torch.save(model.state_dict(), "lstm_snp.pth")