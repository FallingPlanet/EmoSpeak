import torch
import torch.nn as nn

class CustomTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=40, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=768, kernel_size=1)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust shape for Conv1d: [batch, features, seq_len]
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1)  # Permute back to [batch, seq_len, features]

        transformer_output = self.transformer_encoder(x)

        output = transformer_output[:, -1, :]
        output = self.fc(output)

        return output









