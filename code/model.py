import torch
import torch.nn as nn
from torchvision import models

class SignRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(SignRecognitionModel, self).__init__()
        # Use a pre-trained 2D-CNN (e.g., ResNet) as the feature extractor
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer

        # LSTM to capture temporal relationships
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        cnn_out = []

        # Pass each frame through the CNN
        for t in range(seq_length):
            frame = x[:, t, :, :, :]
            cnn_out.append(self.cnn(frame))

        # Stack the CNN outputs and pass through LSTM
        cnn_out = torch.stack(cnn_out, dim=1)
        lstm_out, _ = self.lstm(cnn_out)

        # Take the output of the last LSTM cell
        lstm_out = lstm_out[:, -1, :]

        # Pass through the fully connected layer
        out = self.fc(lstm_out)
        return out