import numpy as np
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """Autoencoder model class."""

    def __init__(self, data: np.ndarray, input_dim: int, hidden_layer_dims: list[int]):
        super(Autoencoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        dataset = TensorDataset(Tensor(data), Tensor(data))
        self.loader = DataLoader(dataset, batch_size=len(data), shuffle=True)

        # Encoder layers
        for i in range(len(hidden_layer_dims)):
            if i == 0:
                self.encoder_layers.append(nn.Linear(input_dim, hidden_layer_dims[i]))
            else:
                self.encoder_layers.append(
                    nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i])
                )
            self.encoder_layers.append(nn.ReLU())

        # Decoder layers
        for i in range(len(hidden_layer_dims) - 1, -1, -1):
            if i == len(hidden_layer_dims) - 1:
                self.decoder_layers.append(nn.Linear(hidden_layer_dims[i], input_dim))
            else:
                self.decoder_layers.append(
                    nn.Linear(hidden_layer_dims[i + 1], hidden_layer_dims[i])
                )
            self.decoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def fit(self, num_epchs=100, lr=0.3):
        criterion = nn.MSELoss()
        # Define the optimizer
        optimizer = optim.SGD(self.parameters(), lr)
        # Train the model
        for _ in range(num_epchs):
            for x, y in self.loader:
                self.forward(x)
                loss = criterion(self.forward(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def forward(self, x) -> np.ndarray:
        """Predict the output of the autoencoder."""

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self) -> np.ndarray:
        """Encode the whole dataset."""

        print("Encoding the data...")
        print("")
        encoded = []
        for x, _ in self.loader:
            encoded.append(self.encoder(x).detach().numpy())
        return np.concatenate(encoded)
