import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomTransformerClassifier(nn.Module):
    def __init__(self, mfcc_dim, spectrogram_dim, num_classes, num_heads=8, num_layers=2, dim_feedforward=1024, dropout=0.1, use_convolutions = True):
        super().__init__()
        combined_input_dim = mfcc_dim + spectrogram_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=combined_input_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)

        # Transformer Encoder Layer
        self.transformer_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )
        
        self.transformer_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )
        

        # Additional layers
        self.layer_norm = nn.LayerNorm(256)
        self.fc = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(dropout)

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

        x = x.permute(2, 0, 1)  # Permute to [seq_len, batch, channels] for Transformer

        # Transformer Encoder
        transformer_output1 = self.transformer_encoder1(x)
        
        transformer_output2 = self.transformer_encoder2(transformer_output1)
        
        

        # Using output of the last token
        output = transformer_output2[-1]

        # Apply layer normalization
        output = self.layer_norm(output)

        # Apply dropout using the dropout layer
        output = self.dropout(output)

        # Fully connected layer for classification
        output = self.fc(output)

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
    










