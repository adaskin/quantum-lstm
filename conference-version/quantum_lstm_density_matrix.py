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
# Global Configuration Parameters
###############################################################################

# Dimensions for the input (system) and hidden (ancilla) registers
n_system_qubits = 2  # Number of system (input) qubits, e.g., Alice
n_ancilla_qubits = (
    2  # Number of ancilla (hidden memory) qubits, e.g., Bob's environment
)
total_qubits = n_system_qubits + n_ancilla_qubits

# Circuit parameters: number of layers and measurement qubit
num_layers = 2  # Number of unitary layers (both for entangling and disentangling)
measurement_qubit = n_system_qubits - 1  # Use the last system qubit for measurement

###############################################################################
# Initialize the PennyLane device.
###############################################################################

dev = qml.device("default.qubit", wires=total_qubits)

###############################################################################
# Quantum Circuit Components
###############################################################################


def entangling_unitary(params, wires):
    """
    Apply an entangling layer with the given parameters.
    'params' is expected to be a tensor of shape (num_layers, len(wires)).
    """
    qml.BasicEntanglerLayers(weights=params, wires=wires, rotation=qml.RY)


def disentangling_unitary(params, wires):
    """
    Apply a disentangling layer in reversed order.
    'params' should be of shape (num_layers, len(wires)); we reverse it along the first dimension.
    """
    reversed_params = torch.flip(params, dims=[0])
    qml.BasicEntanglerLayers(weights=reversed_params, wires=wires, rotation=qml.RY)


@qml.qnode(dev, interface="torch")
def quantum_cell(input_state, h_prev, U_params, Udagger_params):
    """
    Combines the input state and the previous hidden state, embeds them into the circuit,
    applies entangling and disentangling unitaries, and returns:
      - The reduced density matrix on the ancilla (hidden) qubits.
      - An expectation value of Pauli-Z on a configurable system qubit.

    The circuit includes:
      1. Amplitude embedding of the joint state |psi_x> ⊗ |h_prev>.
      2. Circuit evolution via entangling and disentangling unitaries.
      3. Measurement of the ancilla by partial trace to form the density matrix.
      4. A configurable expectation value computed on a system qubit.
    """
    # Combine input and hidden state via the tensor (Kronecker) product.
    joint_state = torch.kron(input_state, h_prev)
    qml.AmplitudeEmbedding(joint_state, wires=range(total_qubits), normalize=True)

    # Apply the entangling and disentangling operations.
    entangling_unitary(U_params, wires=range(total_qubits))
    disentangling_unitary(Udagger_params, wires=range(total_qubits))

    # Return the density matrix on the ancilla qubits and the expectation value measured on a specific qubit.
    return [
        qml.density_matrix(wires=range(n_system_qubits, total_qubits)),
        qml.expval(qml.PauliZ(measurement_qubit)),
    ]


###############################################################################
# Quantum LSTM Implementation
###############################################################################


class QuantumLSTMCell(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the configurable number of layers for U_params.
        self.U_params = nn.Parameter(torch.rand(num_layers, total_qubits))
        self.Udagger_params = nn.Parameter(torch.rand(num_layers, total_qubits))

        # Input projection: maps a scalar x -> a state of dimension 2^(n_system_qubits)
        self.input_proj = nn.Linear(1, 2**n_system_qubits)
        # Output projection: declared for further processing if needed.
        self.output_proj = nn.Linear(2**n_ancilla_qubits, 1)

    def forward(self, x, h_prev):
        """
        x: Tensor of shape (1, 1); one sample input.
        h_prev: Tensor of shape (2**n_ancilla_qubits,); the current hidden state.
        """
        # Project the continuous scalar input to the amplitude space.
        x = x.to(torch.float64).view(-1, 1)  # Ensure shape is (1,1)
        proj_out = self.input_proj(x)  # Shape: (1, 2**n_system_qubits)
        input_state = torch.squeeze(proj_out)  # Now shape: (2**n_system_qubits,)
        input_state = input_state / torch.norm(input_state + 1e-12)

        # Run the quantum circuit.
        h_quantum, output = quantum_cell(
            input_state, h_prev, self.U_params, self.Udagger_params
        )
        # Remove extra batch dimensions if present.
        if h_quantum.dim() == 3:
            h_quantum = h_quantum.squeeze(0)

        # Update the hidden state by taking the diagonal of the reduced density matrix.
        # This effectively uses the probability vector over (ancilla) basis states.
        h_next = torch.diag(torch.real(h_quantum))
        return h_next, output


class QuantumLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = QuantumLSTMCell()
        self.hidden_dim = 2**n_ancilla_qubits  # Dimension of the hidden state.

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, 1) holding the input sequences.
        Processes each sequence sample independently.
        Returns predictions of shape (batch, 1) as output.
        """
        batch_size, seq_len, _ = x.shape
        outputs = []

        for i in range(batch_size):
            # Initialize the hidden state to the basis state |0...0>
            h = torch.zeros(self.hidden_dim, dtype=torch.float64)
            h[0] = 1.0  # Ensure the state is normalized and non-zero.

            out_sample = None
            for t in range(seq_len):
                single_input = x[i, t, :].unsqueeze(0)  # Shape: (1, 1)
                h, out = self.cell(single_input, h)
                out_sample = out  # We use the output of the final time step.
            outputs.append(out_sample.unsqueeze(0))

        return torch.cat(outputs, dim=0).reshape(-1, 1)


if __name__ == "__main__":
    ###############################################################################
    # Training Setup
    ###############################################################################
    # Generate a sine wave dataset for regression.
    def f(x_vals):
        return np.sin(x_vals) + 0.1 * np.random.randn(num_points)

    # Generate a sine wave dataset.
    num_points = 100
    seq_len = 10
    # Generate x values and a noisy sine wave.
    x_vals = np.linspace(0, 8 * math.pi, num_points)
    y_vals = f(x_vals) + 0.1 * np.random.randn(num_points)
    
    # Split the raw data first
    split_idx = int(0.8 * len(y_vals))
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
