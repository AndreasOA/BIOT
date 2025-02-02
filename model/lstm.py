import math
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Convolutional Residual Block 2D
    This block stacks two convolutional layers with batch normalization,
    max pooling, dropout, and residual connection.
    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        stride: stride of the convolutional layers.
        downsample: whether to use a downsampling residual connection.
        pooling: whether to use max pooling.
    Example:
        >>> import torch
        >>> from pyhealth.models import ResBlock2D
        >>>
        >>> model = ResBlock2D(6, 16, 1, True, True)
        >>> input_ = torch.randn((16, 6, 28, 150))  # (batch, channel, height, width)
        >>> output = model(input_)
        >>> output.shape
        torch.Size([16, 16, 14, 75])
    """

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False, pooling=False
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(3, stride=stride, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out
    


class LSTM(nn.Module):
    def __init__(
        self,
        in_channels=16,
        n_classes=6,
        fft=200,
        steps=20,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
        n_segments=5,
    ):
        super().__init__()
        self.fft = fft
        self.steps = steps
        self.n_segments = n_segments
        
        # CNN Feature Extractor
        self.conv1 = ResBlock(in_channels, 32, 2, True, True)
        self.conv2 = ResBlock(32, 64, 2, True, True)
        self.conv3 = ResBlock(64, 128, 2, True, True)
        self.conv4 = ResBlock(128, 256, 2, True, True)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_classes)  # *2 for bidirectional
        )

    def torch_stft(self, x):
        signal = []
        window = torch.hann_window(self.fft, device=x.device)
        for s in range(x.shape[1]):
            spectral = torch.stft(
                x[:, s, :],
                n_fft=self.fft,
                hop_length=self.fft // self.steps,
                win_length=self.fft,
                window=window,
                normalized=True,
                center=True,
                onesided=True,
                return_complex=True,
            )
            signal.append(spectral)
        stacked = torch.stack(signal).permute(1, 0, 2, 3)
        return torch.abs(stacked)

    def cnn(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        # Split input into segments and process through CNN
        n_length = x.shape[2] // self.n_segments
        cnn_emb = [
            self.cnn(x[:, :, idx * n_length : idx * n_length + n_length]).unsqueeze(1)
            for idx in range(self.n_segments)
        ]
        # Concatenate segments: (batch, ts_steps, emb)
        x = torch.cat(cnn_emb, dim=1)
        
        # Process through LSTM
        x, _ = self.lstm(x)
        
        # Global average pooling over time steps
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 16, 2000)
    model = LSTM(in_channels=16, n_classes=6, fft=200, steps=2)
    out = model(x)
    print(out.shape)