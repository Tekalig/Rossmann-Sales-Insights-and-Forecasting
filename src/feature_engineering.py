import pandas as pd
import logging
import joblib
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def extract_date_features(df):
    """Extract date-related features from the 'Date' column."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    logging.info("Date features extracted.")
    return df


def serialize_model(model, path='models/'):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_filename = f"{path}sales_model_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")


class SalesDataset(Dataset):
    def __init__(self, data, target_column):
        self.X = data.drop(columns=[target_column]).values
        self.y = data[target_column].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

def train_lstm(data, target_column, input_dim, hidden_dim, output_dim, epochs=10):
    dataset = SalesDataset(data, target_column)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LSTMModel(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(1))
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model
