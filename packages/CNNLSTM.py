import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(), nn.BatchNorm2d(32))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.BatchNorm2d(64))
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out


class CNN_deltaspec(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN_deltaspec, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(4), nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=1, device="cuda:0"):
        super(LSTM, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # [batch_size, in_channels, n_mfcc, seq_length]
        out = x.squeeze(dim=1)
        in_channels = x.shape[1]
        if in_channels > 1:
            channel_array = []
            for i in range(in_channels):
                channel_data = x[:, i, :, :]
                channel_array.append(channel_data)
            out = torch.cat(channel_array, dim=1)

        # [batch_size, n_mfcc * in_channels, seq_length]
        out = out.permute(0, 2, 1)
        hidden_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        cell_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        # [batch_size, seq_length, n_mfcc * in_channels]
        out, _ = self.lstm(out, (hidden_states, cell_states))
        out = self.flatten(out[:, -1, :])
        return out


class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        n_classes,
        n_layers_rnn=64,
        fc_in=8576,
        in_channels=1,
        device="cuda:0",
        delta_spec=False,
    ):
        super(CNNLSTM, self).__init__()
        if delta_spec:
            self.cnn = CNN_deltaspec(in_channels)
        else:
            self.cnn = CNN(in_channels)

        self.rnn = LSTM(input_size, 64, n_layers_rnn, device=device)
        self.fc1 = nn.Linear(fc_in, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)
        rnn_out = self.rnn(x)
        out = torch.cat([cnn_out, rnn_out], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# model = CNNLSTM(
#     input_size=384,
#     n_classes=2,
#     n_layers_rnn=64,
#     fc_in=3136,
#     device="cpu",
#     in_channels=3,
#     delta_spec=True,
# )
# x = torch.rand([64, 3, 128, 157])

# model.eval()

# # all samples
# ref = model(x)

# # single sample
# out = []
# for x_ in x:
#     x_.unsqueeze_(0)
#     out.append(model(x_))
# out = torch.cat(out)

# print(ref)
# print(out)
