import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerClassifier(nn.Module):
    def __init__(self, mfcc_dim, spectrogram_dim, num_classes, num_heads=8, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        combined_input_dim = mfcc_dim + spectrogram_dim

        # Adjusted convolutional layers
        self.conv1 = nn.Conv1d(in_channels=combined_input_dim, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)

        # Adjust Transformer Encoder Layers (reduced d_model to manage parameter count)
        transformer_layer1 = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)  # Reduced d_model
        transformer_layer2 = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)  # Reduced d_model

        self.transformer_encoder1 = nn.TransformerEncoder(transformer_layer1, num_layers=12)
        self.transformer_encoder2 = nn.TransformerEncoder(transformer_layer2, num_layers=6)

        # Additional layers adjusted for compatibility
        self.layer_norm = nn.LayerNorm(128)  # Adjusted for new d_model
        self.fc1 = nn.Linear(128, 128)  # Adjusted
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)

    # forward method remains the same...


    def forward(self, x_mfcc, x_spectrogram):
        x_combined = torch.cat((x_mfcc, x_spectrogram), dim=1)

        if x_combined.dim() == 4:
            x_combined = x_combined.squeeze(1)

        x = x_combined.permute(0, 2, 1)  # Permute to [batch, seq_len, channels]

        # Convolutional layers with activation and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.permute(2, 0, 1)  # Permute to [seq_len, batch, channels]

        # Transformer Encoder
        x = self.transformer_encoder1(x)
        x = self.transformer_encoder2(x)

        # Using output of the last token
        output = x[-1]

        # Apply layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)

        # Fully connected layers for classification
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        return output




mfcc_dim = 200  # Example dimension for MFCC features
spectrogram_dim = 201  # Example dimension for spectrogram features
num_classes = 6  # Example number of classes

model = CustomTransformerClassifier(mfcc_dim, spectrogram_dim, num_classes)

# Function to calculate the number of trainable parameters in the model
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate and print the number of trainable parameters
trainable_params = count_trainable_parameters(model)
print(f"Trainable Parameters: {trainable_params}")
    










