import torch
import torch.nn as nn
import torch.nn.init as init


class KoopmanNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, device):
        super(KoopmanNetwork, self).__init__()
        self.observable_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.recovery_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.output_dim = output_dim

        # Koopman matrix
        self.k_matrix_raw = nn.Parameter(0.01 * torch.randn(int(output_dim * output_dim)))
        # self.k_matrix = nn.Parameter(0.01 * torch.randn(output_dim, output_dim))  # learnable
        # self.k_matrix = torch.eye(output_dim).to(device)  # fixed

        # Initialize both network parameters using Kaiming normal initialization
        for layer in self.observable_net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        for layer in self.recovery_net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        y = self.observable_net(x)
        y_next = self.predict(y)
        x_next = self.recovery_net(y_next)
        return y, y_next, x_next

    def recover(self, y):
        x_rec = self.recovery_net(y)
        return x_rec

    def observe(self, x):
        y = self.observable_net(x)
        return y

    def predict(self, y):
        k_matrix = self.k_matrix_raw.view(self.output_dim, self.output_dim)
        # k_matrix = self.k_matrix

        y_next = torch.bmm(y.unsqueeze(1), k_matrix.expand(y.size(0), k_matrix.size(0), k_matrix.size(0)))

        return y_next.squeeze(1)  # Remove the extra dimension added by "unsqueeze"
