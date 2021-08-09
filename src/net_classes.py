import torch
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, size, hidden=4096):
        super(LinearNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 200),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.fc(x)



class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 12 * 4, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 200)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.fc(x)
        return output


class LSTMNet(nn.Module):
    # lstm 模型参数的意义，sequence还有layer number
    def __init__(self, input_dim, hidden_dim, n_layers, output_size):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden




