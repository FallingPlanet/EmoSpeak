import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import model as SpeechTransformer
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
from FallingPlanet.orbit.utils.Metrics import TinyEmoBoard
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Classifier:
    def __init__(self, model, device, num_labels, log_dir):
        self.model = model.to(device)
        self.device = device
        self.loss_criterion = nn.CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)

        self.accuracy = Accuracy(num_classes=num_labels, task="multiclass").to(device)
        self.precision = Precision(num_classes=num_labels, task="multiclass").to(device)
        self.recall = Recall(num_classes=num_labels, task="multiclass").to(device)
        self.f1 = F1Score(num_classes=num_labels, task="multiclass").to(device)
        self.mcc = MatthewsCorrCoef(num_classes=num_labels, task="multiclass").to(device)
        self.top2_acc = Accuracy(top_k = 2, num_classes=num_labels, task="multiclass").to(device)

    def train_step(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        train_iterator = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for batch in train_iterator:
            features, labels = batch
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc = self.accuracy(outputs, labels).item()
            train_iterator.set_postfix(loss=total_loss / (train_iterator.n + 1), accuracy=acc)

        train_iterator.close()

    def val_test_step(self, dataloader, phase="Validation"):
        self.model.eval()
        total_loss = 0.0
        iterator = tqdm(dataloader, desc=f"{phase} Epoch")

        with torch.no_grad():
            for batch in iterator:
                features, labels = batch
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)
                loss = self.loss_criterion(outputs, labels)

                total_loss += loss.item()

                # Update metrics
                self.accuracy(outputs, labels)
                self.precision(outputs, labels)
                self.recall(outputs, labels)
                self.f1(outputs, labels)
                self.mcc(outputs, labels)
                self.top2_acc(outputs, labels)

                iterator.set_postfix(loss=total_loss / (iterator.n + 1))

        iterator.close()

        if phase == "Testing":
            final_metrics = {
                "accuracy": self.accuracy.compute().item(),
                "precision": self.precision.compute().item(),
                "recall": self.recall.compute().item(),
                "f1": self.f1.compute().item(),
                "mcc": self.mcc.compute().item(),
                "top 2 accuracy": self.top2_acc.compute().item()
            }

            # Reset metrics
            self.accuracy.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1.reset()
            self.mcc.reset()

            return final_metrics


    




class MFCCDataset(Dataset):
    def __init__(self, mfcc_path, max_length=500):
        self.mfcc_data = torch.load(mfcc_path)
        self.max_length = max_length  # maximum length to pad/truncate to

    def __len__(self):
        return len(self.mfcc_data['labels'])

    def __getitem__(self, idx):
        mfcc = self.mfcc_data['features'][idx]
        label = self.mfcc_data['labels'][idx]

        mfcc = torch.tensor(mfcc, dtype=torch.float32) if not isinstance(mfcc, torch.Tensor) else mfcc
        label = torch.tensor(label, dtype=torch.long) if not isinstance(label, torch.Tensor) else label

        if mfcc.shape[0] > 30:
            mfcc = self.reshape_to_30_channels(mfcc)

        if mfcc.dim() == 3 and mfcc.size(1) == 2:
            mfcc = mfcc.mean(dim=1, keepdim=True)

        # Return the sequence and label as is, without padding here
        return mfcc, label

    def reshape_to_30_channels(self, data):
        # Averaging every two channels for simplicity
        if data.shape[0] > 30:
            data = data.unfold(0, 2, 2).mean(dim=2)
        return data


def custom_collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Ensure all sequences are 3D and have the same size except for the sequence length
    sequences = [seq.unsqueeze(0) if seq.dim() == 2 else seq for seq in sequences]

    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)

    # Stack labels
    labels = torch.stack(labels)

    return sequences_padded, labels


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_path):
        self.spectrogram_data = torch.load(spectrogram_path)
        assert isinstance(self.spectrogram_data['features'][0], torch.Tensor), "Spectrogram data must be tensors."

    def __len__(self):
        return len(self.spectrogram_data['labels'])

    def __getitem__(self, idx):
        spectrogram = self.spectrogram_data['features'][idx]
        label = self.spectrogram_data['labels'][idx]

        # Ensure both spectrogram and label are tensors
        if not isinstance(spectrogram, torch.Tensor):
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return spectrogram, label





feature_type = 'mfcc'  # Change to 'spectrogram' to use spectrogram features

# Paths to your datasets
mfcc_train_path = r"E:\speech_datasets\FP_EmoSpeak_MFCCs_Train.pt"
mfcc_val_path = r"E:\speech_datasets\FP_EmoSpeak_MFCCs_Val.pt"
mfcc_test_path = r"E:\speech_datasets\FP_EmoSpeak_MFCCs_Test.pt"
spectrogram_train_path = r"E:\speech_datasets\FP_EmoSpeak_Spectrograms_Train.pt"
spectrogram_val_path = r"E:\speech_datasets\FP_EmoSpeak_Spectrograms_Val.pt"
spectrogram_test_path = r"E:\speech_datasets\FP_EmoSpeak_Spectrograms_Test.pt"

# Set the batch size for the DataLoader
batch_size = 1  # You can adjust this based on your requirements and system capabilities

# Initialize the dataset and DataLoader based on the selected feature type
if feature_type == 'mfcc':
    train_dataset = MFCCDataset(mfcc_train_path)
    val_dataset = MFCCDataset(mfcc_val_path)
    test_dataset = MFCCDataset(mfcc_test_path)
    feature_dim = 30  # Set the correct MFCC dimension
elif feature_type == 'spectrogram':
    train_dataset = SpectrogramDataset(spectrogram_train_path)
    val_dataset = SpectrogramDataset(spectrogram_val_path)
    test_dataset = SpectrogramDataset(spectrogram_test_path)
    feature_dim = 214  # Set the correct Spectrogram dimension

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7  # Number of classes for your classification task

# Initialize the model
model = SpeechTransformer.CustomTransformerClassifier(
    target_channels=feature_dim, 
    num_classes=num_classes, 
    num_heads=16, 
    dim_feedforward=2048, 
    num_layers=4, 
    dropout=0.1
)

# Initialize the Classifier
classifier = Classifier(
    model=model,
    device=device,
    num_labels=num_classes,
    log_dir="logs"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    classifier.train_step(train_loader, optimizer, epoch)
    classifier.val_test_step(val_loader, "Validation")

# Testing
test_results = classifier.val_test_step(test_loader, "Testing")

# Print test results


print("\nTest Metrics:")
print(f"Accuracy: {test_results['accuracy']:.4f}")
print(f"Precision: {test_results['precision']:.4f}")
print(f"Recall: {test_results['recall']:.4f}")
print(f"F1 Score: {test_results['f1']:.4f}")
print(f"Matthews Correlation Coefficient: {test_results['mcc']:.4f}")
print(f"Top 2 Accuracy: {test_results['top 2 accuracy']:.4f}")

model_save_path = r"E:\model_saves\EmoSpeak_Transformer_Tiny.pt"  # Adjust the path and filename as needed

# Check if the directory exists, if not, create it
model_save_dir = os.path.dirname(model_save_path)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")