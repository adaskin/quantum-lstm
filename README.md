### Quantum RNNs and LSTMs Through Entangling and Disentangling Power of Unitary Transformations, Ammar Daskin, 2025. 
 [https://arxiv.org/pdf/2505.06774](https://arxiv.org/pdf/2505.06774)

The simulation code that was used to generate the figures in the paper:
- *Quantum RNNs and LSTMs Through Entangling and Disentangling Power of Unitary Transformations, Ammar Daskin, 2025. [https://arxiv.org/pdf/2505.06774](https://arxiv.org/pdf/2505.06774)*

You can run `quantum_lstm_collapsed_state.py` or `quantum_lstm_density_matrix.py` as indicated by their names one of them uses the normalized collapsed state as the hidden state $|h_t \rangle$, the other one uses the probabilities, where we use the diagonal of density matrix, this can be found without generating density matrix.

The code is mostly self-explanatory. However, if you want to run this on different data, you can use `QuantumLSTM()` from the above modules to run on your data.

For this 
- just import;

    ```python
    from quantum_lstm_collapsed_state import QuantumLSTM
    ```

    and then you can use/modify the train_model function
    ```python
    train_model(
        model, optimizer, loss_fn, X_train, y_train, num_epochs=50, batch_size=5
    )
    ```
- Or you can just simply modify the definition of noisy sin function `f(x_vals)` we have used in `quantum_lstm_*.py` files.


Note that the model is using very basic parameterized circuits for entangling gates. 
For complex problems, you can use more layered circuits in entangling and sientangling functions or increase the number of qubits given as global parameters in  `quantum_lstm_*.py` files to increase number of trainable paramaters.
