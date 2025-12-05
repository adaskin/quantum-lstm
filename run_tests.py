# This is a script to run tests on the Quantum LSTM model.
from quantum_lstm_collapsed_state import QuantumLSTM

# Uncomment the following line if you want to use the density matrix version
#from quantum_lstm_density_matrix import QuantumLSTM
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler


def generate_recent_weather_data(seq_len=10, num_days=30, train_split=0.8):
    """
    Retrieve recent weather data (e.g., last num_days days) for a selected region,
    create input sequences for forecasting, and split the data into training and test sets.

    Args:
        seq_len (int): Length of each sequence (number of time steps per sample).
        num_days (int): Number of past days to include. (Smaller num_days → fewer data points.)
        train_split (float): Fraction of data used for training.

    Returns:
        X_train (torch.Tensor): Training sequences with shape (samples, seq_len, 1).
        y_train (torch.Tensor): Training targets with shape (samples, 1).
        X_test (torch.Tensor): Test sequences.
        y_test (torch.Tensor): Test targets.
        scaler (MinMaxScaler): Scaler fitted to the temperature data (for inverse transformations).
    """
    from meteostat import Stations, Daily

    # Find weather stations in Istanbul (region code "34" in Turkey)
    from meteostat import Stations  # local import in case Meteostat isn't always used

    stations = Stations()
    stations = stations.region("CA", "ON")  # Canada, Ontario region
    station_df = stations.fetch(1)  # Get one station
    if station_df.empty:
        raise ValueError("No station found for the specified region.")
    station_id = station_df.index[0]

    # Define the time period: get data for the past num_days
    end = datetime.now()
    start = end - timedelta(days=num_days)

    # Get daily data for the selected station
    data = Daily(station_id, start, end).fetch()
    data.reset_index(inplace=True)  # Make sure date is a column

    # Choose temperature metric: use 'tavg' if available, otherwise 'tmin'
    if "tavg" in data.columns and not data["tavg"].isnull().all():
        temps = data["tavg"]
    else:
        temps = data["tmin"]

    # Drop missing values and convert to a numpy array
    temps = temps.dropna().values.reshape(-1, 1)
    
    # Split the data BEFORE fitting scaler
    split_idx = int(train_split * len(temps))
    train_temps = temps[:split_idx]
    test_temps = temps[split_idx:]

    # Normalize the temperature values using MinMaxScaler - FIT ONLY ON TRAINING
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_temps).flatten()
    test_scaled = scaler.transform(test_temps).flatten()
    
    # Combine scaled data for sequencing (test data not used in fitting)
    temps_scaled = np.concatenate([train_scaled, test_scaled])

    # Create sequences (rolling windows)
    X_seq, y_seq = [], []
    for i in range(len(temps_scaled) - seq_len):
        X_seq.append(temps_scaled[i : i + seq_len])
        y_seq.append(temps_scaled[i + seq_len])

    X_seq = np.array(X_seq)  # Shape: (num_samples, seq_len)
    y_seq = np.array(y_seq).reshape(-1, 1)  # Shape: (num_samples, 1)

    # Convert to torch tensors. Add a channel dimension so each sequence element is a 1D vector.
    X_tensor = torch.tensor(X_seq, dtype=torch.float64).unsqueeze(-1)
    y_tensor = torch.tensor(y_seq, dtype=torch.float64)

    # Split into training (first train_split fraction) and test sets.
    split_idx = int(train_split * X_tensor.size(0))
    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]
    X_test = X_tensor[split_idx:]
    y_test = y_tensor[split_idx:]

    return X_train, y_train, X_test, y_test, scaler


def generate_weather_data(csv_path, seq_len=10, train_split=0.8):
    """
    Load weather data from a CSV file, create input sequences for forecasting,
    and split the data into training and test sets.

    Args:
        csv_path (str): Path or URL to the CSV file containing weather data.
                        The CSV should include columns 'Date' and 'Temp'.
        seq_len (int): Length of each input sequence (number of time steps).
        train_split (float): Fraction of data to use for training.

    Returns:
        X_train (torch.Tensor): Training input sequences with shape (samples, seq_len, 1).
        y_train (torch.Tensor): Training targets with shape (samples, 1).
        X_test  (torch.Tensor): Test input sequences.
        y_test  (torch.Tensor): Test targets.
        scaler  (MinMaxScaler): The scaler fitted to the temperature data (for inverse transform).
    """
    # Load CSV data; if using a URL, csv_path can be the URL.
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Sort by date so the sequence is sequential.
    df.sort_values("Date", inplace=True)

    # Extract the temperature data.
    temps = df["Temp"].values.reshape(-1, 1)  # Ensure 2D array for scaler
    
    # Split the data BEFORE fitting scaler
    split_idx = int(train_split * len(temps))
    train_temps = temps[:split_idx]
    test_temps = temps[split_idx:]

    # Normalize temperatures between 0 and 1 - FIT ONLY ON TRAINING
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_temps).flatten()
    test_scaled = scaler.transform(test_temps).flatten()
    
    # Combine scaled data for sequencing
    temps_scaled = np.concatenate([train_scaled, test_scaled])

    # Create sequences: each sequence of length seq_len has the next time step as target.
    X_seq, y_seq = [], []
    for i in range(len(temps_scaled) - seq_len):
        X_seq.append(temps_scaled[i : i + seq_len])
        y_seq.append(temps_scaled[i + seq_len])

    X_seq = np.array(X_seq)  # Shape: (num_samples, seq_len)
    y_seq = np.array(y_seq).reshape(-1, 1)  # Shape: (num_samples, 1)

    # Convert the NumPy arrays to PyTorch tensors.
    # The input tensor has an added channel dimension (last dimension) for compatibility.
    X_tensor = torch.tensor(X_seq, dtype=torch.float64).unsqueeze(-1)
    y_tensor = torch.tensor(y_seq, dtype=torch.float64)

    # Split the data into training and test sets.
    split_idx = int(train_split * X_tensor.size(0))
    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]
    X_test = X_tensor[split_idx:]
    y_test = y_tensor[split_idx:]

    return X_train, y_train, X_test, y_test, scaler


def generate_stock_data(csv_path, seq_len=10, feature="Close", train_split=0.8):
    """
    Load stock market data from a CSV file, create input sequences for time-series forecasting,
    and split the data into training and test sets.

    Args:
        csv_path (str): Path to the CSV file containing stock market data. The CSV should include
                        a 'Date' column and a column for the desired feature to forecast (e.g., "Close").
        seq_len (int): Length of each input sequence.
        feature (str): The column name to use as the target (e.g., "Close").
        train_split (float): Fraction of data to use for training (rest for testing).

    Returns:
        X_train (torch.Tensor): Training input sequences of shape (samples, seq_len, 1).
        y_train (torch.Tensor): Training targets of shape (samples, 1).
        X_test  (torch.Tensor): Test input sequences.
        y_test  (torch.Tensor): Test targets.
        scaler  (MinMaxScaler): The scaler object applied (for inverse transformation later).
    """
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Ensure data is sorted by date (assuming CSV has a 'Date' column)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)

    # Extract the desired feature:
    prices = df[feature].values.reshape(-1, 1)
    
    # Split the data BEFORE fitting scaler
    split_idx = int(train_split * len(prices))
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]

    # Normalize the data with MinMaxScaler - FIT ONLY ON TRAINING
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices).flatten()
    test_scaled = scaler.transform(test_prices).flatten()
    
    # Combine scaled data for sequencing
    prices_scaled = np.concatenate([train_scaled, test_scaled])

    # Create sequences: each sequence of length seq_len with the target as the next data point.
    X_seq = []
    y_seq = []
    for i in range(len(prices_scaled) - seq_len):
        X_seq.append(prices_scaled[i : i + seq_len])
        y_seq.append(prices_scaled[i + seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq).reshape(-1, 1)

    # Convert sequences to torch tensors.
    # X_tensor shape: (samples, seq_len, 1); y_tensor shape: (samples, 1)
    X_tensor = torch.tensor(X_seq, dtype=torch.float64).unsqueeze(-1)
    y_tensor = torch.tensor(y_seq, dtype=torch.float64)

    # Split data into training and test sets.
    split_idx = int(train_split * X_tensor.size(0))
    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]
    X_test = X_tensor[split_idx:]
    y_test = y_tensor[split_idx:]

    return X_train, y_train, X_test, y_test, scaler


###############################################################################
# Data Generation Function
###############################################################################
def generate_sin_data(num_points=100, seq_len=10, noise_std=0.1, train_split=0.8):
    """
    Generate a noisy sine wave dataset, create sequence inputs for a regression task,
    and split into training and test sets.

    Args:
        num_points (int): Total number of data points.
        seq_len (int): Length of each input sequence.
        noise_std (float): Standard deviation of added Gaussian noise.
        train_split (float): Fraction of data used for training.

    Returns:
        X_train (torch.Tensor): Training input sequences of shape (samples, seq_len, 1).
        y_train (torch.Tensor): Training targets of shape (samples, 1).
        X_test (torch.Tensor): Test input sequences.
        y_test (torch.Tensor): Test targets.
    """
    # Generate x values and a noisy sine wave.
    x_vals = np.linspace(0, 8 * math.pi, num_points)
    y_vals = np.sin(x_vals) + noise_std * np.random.randn(num_points)
    
    # Split the raw data first
    split_idx = int(train_split * len(y_vals))
    train_vals = y_vals[:split_idx]
    test_vals = y_vals[split_idx:]
    
    # Create sequences separately for train and test
    X_train_seq, y_train_seq = [], []
    for i in range(len(train_vals) - seq_len):
        X_train_seq.append(train_vals[i : i + seq_len])
        y_train_seq.append(train_vals[i + seq_len])
    
    X_test_seq, y_test_seq = [], []
    for i in range(len(test_vals) - seq_len):
        X_test_seq.append(test_vals[i : i + seq_len])
        y_test_seq.append(test_vals[i + seq_len])

    # Convert to torch tensors
    X_train = torch.tensor(np.array(X_train_seq), dtype=torch.float64).unsqueeze(-1)
    y_train = torch.tensor(np.array(y_train_seq), dtype=torch.float64).reshape(-1, 1)
    X_test = torch.tensor(np.array(X_test_seq), dtype=torch.float64).unsqueeze(-1)
    y_test = torch.tensor(np.array(y_test_seq), dtype=torch.float64).reshape(-1, 1)

    return X_train, y_train, X_test, y_test


###############################################################################
# Training Function
###############################################################################
def train_model(
    model, optimizer, loss_fn, X_train, y_train, num_epochs=50, batch_size=5
):
    """
    Train the given model on the training data.

    Args:
        model (nn.Module): The neural (quantum LSTM) model.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        X_train (torch.Tensor): Training inputs.
        y_train (torch.Tensor): Training targets.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Mini-batch size.

    Returns:
        train_losses (list): List of average loss per epoch.
    """
    train_losses = []
    for epoch in range(num_epochs):
        # Shuffle training data indices.
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0

        # Process data in mini-batches.
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            predictions = model(batch_x)  # Expected shape: (batch, 1)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / X_train.size(0)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return train_losses


###############################################################################
# Plotting Functions
###############################################################################
def plot_training_loss(train_losses, title="Training Loss"):
    """
    Plot the training loss curve.

    Args:
        train_losses (list): List of per-epoch training losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(
    model,
    X_test,
    y_test,
    xlabel="Sample Index",
    ylabel="Value",
    title="Quantum LSTM: Predictions vs. True Values",
):
    """
    Evaluate the model on the test set and plot predictions versus true values.

    Args:
        model (nn.Module): Trained model.
        X_test (torch.Tensor): Test inputs.
        y_test (torch.Tensor): True targets.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert predictions and targets to numpy arrays.
    y_pred_np = y_pred.squeeze().numpy()
    y_test_np = y_test.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(y_pred_np, label="Predictions", linestyle="-", marker="*")
    plt.plot(y_test_np, label="True Values", linestyle="--", marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def run_weather(num_epochs=50, batch_size=5, seq_len=10, train_split=0.8):
    # Example usage:
    csv_path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    X_train, y_train, X_test, y_test, scaler = generate_weather_data(
        csv_path, seq_len=seq_len, train_split=train_split
    )

    model = QuantumLSTM()  # Your QuantumLSTM model class
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_losses = train_model(
        model,
        optimizer,
        loss_fn,
        X_train,
        y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    plot_training_loss(train_losses)
    evaluate_model(model, X_test, y_test)


def run_recent_weather(
    num_epochs=50, batch_size=5, seq_len=10, train_split=0.8, num_days=30
):

    X_train, y_train, X_test, y_test, scaler = generate_recent_weather_data(
        num_days=num_days, seq_len=seq_len, train_split=train_split
    )

    # Now continue with training using your QuantumLSTM model.

    model = QuantumLSTM()  # Your QuantumLSTM model class
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_losses = train_model(
        model,
        optimizer,
        loss_fn,
        X_train,
        y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    plot_training_loss(train_losses)
    evaluate_model(model, X_test, y_test)


def run_sin(
    num_epochs=50,
    batch_size=5,
    num_points=100,
    seq_len=5,
    noise_std=0.1,
    train_split=0.8,
):
    # Generate the dataset.
    X_train, y_train, X_test, y_test = generate_sin_data(
        num_points=num_points,
        seq_len=seq_len,
        noise_std=noise_std,
        train_split=train_split,
    )

    # Define training hyperparameters.
    num_epochs = 50
    batch_size = 5

    # Initialize the model, optimizer, and loss function.
    model = QuantumLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Train the model.
    train_losses = train_model(
        model,
        optimizer,
        loss_fn,
        X_train,
        y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    # Plot training loss.
    plot_training_loss(train_losses)

    # Evaluate and plot test predictions.
    evaluate_model(model, X_test, y_test)


###############################################################################
# Example Main Script Using the Modular Functions
###############################################################################
if __name__ == "__main__":
    # Assume QuantumLSTM (and its components) is already defined and imported.
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 14,
            "figure.figsize": (10, 5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )
    # Uncomment the following lines to run with different datasets.
    # run_weather(num_epochs = 15, batch_size = 5)
    # run_sin(num_epochs = 150, batch_size = 5,train_split=0.8)

    # run_stock(csv_path="path_to_your_stock_data.csv", seq_len=10, feature="Close", train_split=0.8)
    # run_weather(csv_path="path_to_your_weather_data.csv", seq_len=10, train_split=0.8)
    run_recent_weather(
        seq_len=10, num_days=365, batch_size=5, num_epochs=100, train_split=0.8
    )
