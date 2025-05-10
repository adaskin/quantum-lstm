###############################################################################
# A Quantum LSTM that use ancilla for hidden state.
# Example Usage (Training, Evaluation, etc. would be)
###############################################################################

# For instance, you may  instantiate your model:
# model = QuantumLSTM()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# etc.
import pennylane as qml
from pennylane import numpy as np  # PennyLane’s NumPy wrapper.
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

# Set seed for reproducibility and use double precision.
torch.manual_seed(123)
torch.set_default_dtype(torch.float64)
np.random.seed(123)

###############################################################################
# Quantum Circuit Configuration (Fixed Wire Handling)
###############################################################################

# Number of qubits in the input (system) and hidden (ancilla) parts.
n_system_qubits = 2  # Input system qubits (Alice)
n_ancilla_qubits = 2  # Hidden state qubits (Bob's environment)
total_qubits = n_system_qubits + n_ancilla_qubits

# Circuit and measurement parameters.
num_layers = 2  # Number of entangling layers for the circuit (configurable)
measurement_qubit = (
    n_system_qubits - 1
)  # Use the last of the system qubits for measurement

# Initialize the PennyLane device.
dev = qml.device("default.qubit", wires=total_qubits)


def entangling_unitary(params, wires):
    """Apply an entangling layer with the given parameters."""
    qml.BasicEntanglerLayers(weights=params, wires=wires, rotation=qml.RY)


def disentangling_unitary(params, wires):
    """Apply a disentangling layer in reversed order using torch.flip."""
    reversed_params = torch.flip(params, dims=[0])
    qml.BasicEntanglerLayers(weights=reversed_params, wires=wires, rotation=qml.RY)


@qml.qnode(dev, interface="torch")
def quantum_cell(input_state, h_prev, U_params, Udagger_params):
    """
    Combines the input state and the previous hidden state, embeds them into the circuit,
    applies entangling and disentangling unitaries, and returns:
      - The full joint state vector as produced by the circuit.
      - An expectation value measured on a configurable system qubit.
    """
    # Combine input and hidden state via tensor (Kronecker) product.
    joint_state = torch.kron(input_state, h_prev)
    qml.AmplitudeEmbedding(joint_state, wires=range(total_qubits), normalize=True)

    # Apply the unitary operations.
    entangling_unitary(U_params, wires=range(total_qubits))
    disentangling_unitary(Udagger_params, wires=range(total_qubits))

    # Return the full state vector and the expectation value measured on `measurement_qubit`
    return [qml.state(), qml.expval(qml.PauliZ(measurement_qubit))]


###############################################################################
# Quantum LSTM Implementation (Adaptable Parameter Handling)
###############################################################################


class QuantumLSTMCell(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the configurable number of layers.
        self.U_params = nn.Parameter(torch.rand(num_layers, total_qubits))
        self.Udagger_params = nn.Parameter(torch.rand(num_layers, total_qubits))

        # Input projection: maps a scalar to a state of size 2^(n_system_qubits).
        self.input_proj = nn.Linear(1, 2**n_system_qubits)
        # Output projection (declared for potential further processing).
        self.output_proj = nn.Linear(2**n_ancilla_qubits, 1)

    def forward(self, x, h_prev):
        """
        x: Tensor of shape (1, 1) containing one sample input.
        h_prev: Tensor of shape (2**n_ancilla_qubits,) representing the current hidden state.
        """
        # Ensure correct dtype and shape.
        x = x.to(torch.float64).view(-1, 1)  # Shape: (1, 1)
        proj_out = self.input_proj(x)  # Shape: (1, 2**n_system_qubits)
        input_state = torch.squeeze(proj_out)  # Now shape: (2**n_system_qubits,)
        input_state = input_state / torch.norm(
            input_state + 1e-12
        )  # Avoid division by zero

        # Run the quantum circuit.
        joint_state, output = quantum_cell(
            input_state, h_prev, self.U_params, self.Udagger_params
        )
        # 'joint_state' is a vector of length 2^(total_qubits).
        # Reshape it so that rows correspond to outcomes on the system (input) qubits
        # and columns correspond to the hidden (ancilla) qubits.
        state_reshaped = joint_state.reshape(2**n_system_qubits, 2**n_ancilla_qubits)

        # Compute the probabilities for each outcome of the system qubits.
        row_probs = torch.sum(torch.abs(state_reshaped) ** 2, dim=1)

        # Determine the outcome with the highest probability.
        max_idx = torch.argmax(row_probs)
        # Extract the corresponding branch (the collapsed hidden state).
        collapsed_hidden = state_reshaped[max_idx, :]

        # Normalize the collapsed state.
        h_next = collapsed_hidden / torch.norm(collapsed_hidden + 1e-12)

        return h_next, output


class QuantumLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = QuantumLSTMCell()
        self.hidden_dim = 2**n_ancilla_qubits

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, 1) containing the input sequences.
        Processes each sample in the batch independently.
        Returns predictions of shape (batch, 1).
        """
        batch_size, seq_len, _ = x.shape
        outputs = []

        for i in range(batch_size):
            # Initialize hidden state as a valid quantum basis state |0...0>
            h = torch.zeros(self.hidden_dim, dtype=torch.float64)
            h[0] = 1.0  # Set the first element to 1

            out_sample = None
            for t in range(seq_len):
                single_input = x[i, t, :].unsqueeze(0)  # Shape: (1, 1)
                h, out = self.cell(single_input, h)
                out_sample = out  # Use the output of the final time step as prediction.
            outputs.append(out_sample.unsqueeze(0))

        return torch.cat(outputs, dim=0).reshape(-1, 1)


if __name__ == "__main__":
    ###############################################################################
    # Training Setup
    ###############################################################################
    def f(x_vals):
        return np.sin(x_vals) + 0.1 * np.random.randn(num_points)

    # Generate a sine wave dataset.
    num_points = 100
    seq_len = 10

    x_vals = np.linspace(0, 8 * math.pi, num_points)
    y_vals = f(x_vals)  # Shape: (num_points,)

    # Create sequences: each input is a sequence of length seq_len, and the target is the next value.
    X_seq = np.array([y_vals[i : i + seq_len] for i in range(num_points - seq_len)])
    y_seq = np.array(y_vals[seq_len:]).reshape(-1, 1)

    # Convert to torch tensors.
    X_tensor = torch.tensor(X_seq, dtype=torch.float64).unsqueeze(
        -1
    )  # Shape: (samples, seq_len, 1)
    y_tensor = torch.tensor(y_seq, dtype=torch.float64)  # Shape: (samples, 1)

    # Split the data into training (80%) and test (20%) sets.
    split_idx = int(0.8 * X_tensor.size(0))
    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]
    X_test = X_tensor[split_idx:]
    y_test = y_tensor[split_idx:]

    # Training hyperparameters.
    num_epochs = 150
    batch_size = 5
    train_losses = []

    # Initialize the model, optimizer, and loss function.
    model = QuantumLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop.
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            optimizer.zero_grad()
            pred = model(batch_x)  # Expected shape: (batch, 1)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / X_train.size(0)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

    ###############################################################################
    # Plotting the training loss curve.
    ###############################################################################
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
    # Plot the training loss curve.
    # plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss on Sine Regression Task")
    plt.legend()
    plt.grid(True)
    plt.show()

    ###############################################################################
    # Test Evaluation and Plotting
    ###############################################################################

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert predictions and targets to numpy arrays.
    y_pred_np = y_pred.squeeze().numpy()
    y_test_np = y_test.squeeze().numpy()

    # plt.figure(figsize=(10, 4))
    plt.plot(y_pred_np, label="Predictions", linestyle="-", marker="*")
    plt.plot(y_test_np, label="True Values", linestyle="--", marker="o")
    plt.xlabel("Sample Index")
    plt.ylabel("Sine Value")
    plt.title("Quantum LSTM: Predictions vs. True Values on Sine Wave")
    plt.legend()
    plt.grid(True)
    plt.show()
