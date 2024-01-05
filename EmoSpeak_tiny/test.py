import torch
import torch.nn as nn

class CustomTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerClassifier, self).__init__()

       

        # Transformer Encoder Layer
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_layers)
        
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward,dropout=dropout)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2,num_layers=num_layers)
        
        self.dense1 = nn.Linear(input_dim,256)
        self.dense2 = nn.Linear(256,128)
        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        

        # Passing the input through the Transformer layers
        transformer_output1 = self.transformer_encoder1(x)
        
        transformer_output2 = self.transformer_encoder2(transformer_output1)

        # We use the output of the last token for classification
        output = transformer_output2[:, -1, :]

        # Passing the output through the fully connected layer
        output = self.fc(output)

        return output

# Example parameters
input_dim = 768  # Feature size
num_heads = 8
num_layers = 4
num_classes = 10  # Number of emotion classes

# Initialize the model
model = CustomTransformerClassifier(input_dim, num_heads, num_layers, num_classes)

# Calculate the number of trainable parameters
num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in the model: {num_trainable_parameters}")


