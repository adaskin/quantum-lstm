###############################################################################
# Configurable Quantum LSTM with angle embedding (non‑linear encoding)
# n_system_qubits and n_ancilla_qubits are set when creating the cell.
# Each input sequence is processed independently (the hidden state is reset
# to |0...0> at the beginning of every sequence). This is NOT a stateful model
# across different windows.
###############################################################################
import pennylane as qml
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from pennylane import numpy as np

torch.manual_seed(123)
torch.set_default_dtype(torch.float64)
np.random.seed(123)


class QuantumLSTMCell(nn.Module):
    def __init__(self, n_system_qubits=2, n_ancilla_qubits=2, num_layers=2):
        super().__init__()
        self.n_sys = n_system_qubits
        self.n_anc = n_ancilla_qubits
        self.total_qubits = n_system_qubits + n_ancilla_qubits
        self.num_layers = num_layers
        self.measurement_qubit = n_system_qubits - 1  # last system qubit for output

        # Device: using default.qubit (can be changed to lightning.qubit later)
        self.dev = qml.device("default.qubit", wires=self.total_qubits)

        # Define the quantum node (captures self)
        @qml.qnode(self.dev, interface="torch")
        def quantum_cell(input_features, h_prev, U_params, Udagger_params):
            """
            input_features: (n_sys,) tensor – one scalar per system qubit
            h_prev: (2**n_anc,) tensor – amplitude vector for ancilla
            """
            # --- Encode input on system qubits with non‑linear rotations ---
            # --- Cyclic encoding of input features onto system qubits ---
            n_features = input_features.shape[0]
      
            for qubit in range(self.n_sys):
                f_indx = qubit % n_features   # round‑robin over features
                f = input_features[f_indx]
                f = torch.tanh(f)   # safe range for sqrt [-1, 1]
                qml.RY(np.pi * f, wires=qubit)
                qml.RZ(np.pi * (f ** 3), wires=qubit)
                qml.RX(np.pi * torch.sqrt(1 - f**2), wires=qubit)

            # --- Encode hidden state on ancilla qubits via amplitude embedding ---
            qml.AmplitudeEmbedding(h_prev, wires=range(self.n_sys, self.total_qubits), normalize=True)

            # --- Entangling and disentangling unitaries ---
            # BasicEntanglerLayers expects params (num_layers, total_qubits)
            qml.BasicEntanglerLayers(weights=U_params, wires=range(self.total_qubits), rotation=qml.RY)
            # For disentangling, we reverse the order of layers (flip first dimension)
            reversed_params = torch.flip(Udagger_params, dims=[0])
            qml.BasicEntanglerLayers(weights=reversed_params, wires=range(self.total_qubits), rotation=qml.RY)

            # Return full state vector and the expectation value on the measurement qubit
            return qml.state(), qml.expval(qml.PauliZ(self.measurement_qubit))

        self.quantum_cell = quantum_cell

        # Trainable parameters for the unitaries
        self.U_params = nn.Parameter(torch.rand(self.num_layers, self.total_qubits))
        self.Udagger_params = nn.Parameter(torch.rand(self.num_layers, self.total_qubits))

        # Input projection: scalar -> n_sys features (one per system qubit)
        self.input_proj = nn.Linear(1, self.n_sys)

    def forward(self, x, h_prev):
        """
        x: scalar input, tensor shape (1, 1)
        h_prev: previous hidden state, shape (2**n_anc,)
        Returns: next hidden state (same shape), output (scalar prediction)
        """
        # In forward(), replace the projection with:
        #x_val = x.item()   # scalar
        features = x #torch.tensor(x, dtype=torch.float64)

        # Run quantum circuit
        joint_state, output = self.quantum_cell(
            features, h_prev, self.U_params, self.Udagger_params
        )
        # joint_state shape: (2**total_qubits,)

        # Reshape to separate system and ancilla basis
        state_reshaped = joint_state.reshape(2**self.n_sys, 2**self.n_anc)

        # Collapse: pick the most probable system basis state, take the corresponding ancilla branch
        row_probs = torch.sum(torch.abs(state_reshaped) ** 2, dim=1)
        max_idx = torch.argmax(row_probs)
        collapsed_hidden = state_reshaped[max_idx, :]
        h_next = collapsed_hidden / (torch.norm(collapsed_hidden) + 1e-12)

        return h_next, output


class QuantumLSTM(nn.Module):
    def __init__(self, n_system_qubits=2, 
                 n_ancilla_qubits=2, 
                 num_layers=4):
        super().__init__()
        self.cell = QuantumLSTMCell(n_system_qubits, n_ancilla_qubits, num_layers)
        self.hidden_dim = 2 ** n_ancilla_qubits

    def forward(self, x):
        """
        x: shape (batch, seq_len, 1)
        Returns: predictions shape (batch, 1)
        """
        batch_size, seq_len, _ = x.shape
        outputs = []

        for i in range(batch_size):
            # Initialize hidden state to |0...0> (basis state)
            h = torch.zeros(self.hidden_dim, dtype=torch.float64)
            h[0] = 1.0

            out_sample = None
            for t in range(seq_len):
                single_input = x[i, t, :].unsqueeze(0)   # (1,1)
                h, out = self.cell(single_input, h)
                out_sample = out   # take output from the last time step
            outputs.append(out_sample.unsqueeze(0))

        return torch.cat(outputs, dim=0).reshape(-1, 1)


# ----------------------------------------------------------------------
# Example: training on noisy sine (same as before, but now configurable)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters for the quantum LSTM
    N_SYSTEM_QUBITS = 2
    N_ANCILLA_QUBITS = 2
    NUM_LAYERS = 2

    # Generate data (noisy sine)
    num_points = 100
    seq_len = 10
    x_vals = np.linspace(0, 8 * math.pi, num_points)
    y_vals = np.sin(x_vals) + 0.1 * np.random.randn(num_points)

    # Split and create sequences
    split_idx = int(0.8 * len(y_vals))
    train_vals = y_vals[:split_idx]
    test_vals = y_vals[split_idx:]

    X_train_seq, y_train_seq = [], []
    for i in range(len(train_vals) - seq_len):
        X_train_seq.append(train_vals[i:i+seq_len])
        y_train_seq.append(train_vals[i+seq_len])

    X_test_seq, y_test_seq = [], []
    for i in range(len(test_vals) - seq_len):
        X_test_seq.append(test_vals[i:i+seq_len])
        y_test_seq.append(test_vals[i+seq_len])

    X_train = torch.tensor(np.array(X_train_seq), dtype=torch.float64).unsqueeze(-1)
    y_train = torch.tensor(np.array(y_train_seq), dtype=torch.float64).reshape(-1, 1)
    X_test = torch.tensor(np.array(X_test_seq), dtype=torch.float64).unsqueeze(-1)
    y_test = torch.tensor(np.array(y_test_seq), dtype=torch.float64).reshape(-1, 1)

    # Model, optimizer, loss
    model = QuantumLSTM(N_SYSTEM_QUBITS, N_ANCILLA_QUBITS, NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    num_epochs = 150
    batch_size = 5
    train_losses = []

    for epoch in range(num_epochs):
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / X_train.size(0)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss on Sine Regression Task")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate and plot predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    y_pred_np = y_pred.squeeze().numpy()
    y_test_np = y_test.squeeze().numpy()
    plt.plot(y_pred_np, label="Predictions", linestyle="-", marker="*")
    plt.plot(y_test_np, label="True Values", linestyle="--", marker="o")
    plt.xlabel("Sample Index")
    plt.ylabel("Sine Value")
    plt.title("Quantum LSTM: Predictions vs. True Values on Sine Wave")
    plt.legend()
    plt.grid(True)
    plt.show()