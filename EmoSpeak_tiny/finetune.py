import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import model as SpeechTransformer
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
from FallingPlanet.orbit.utils.Metrics import TinyEmoBoard
import os

class Classifier:
    def __init__(self, model, device, num_labels, log_dir, mfcc_dim, spectrogram_dim):
        self.model = model.to(device)
        self.device = device
        self.mfcc_dim = mfcc_dim
        self.spectrogram_dim = spectrogram_dim
        self.loss_criterion = nn.CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)

        self.accuracy = Accuracy(num_classes=num_labels, task="multiclass").to(device)
        self.precision = Precision(num_classes=num_labels, task="multiclass").to(device)
        self.recall = Recall(num_classes=num_labels, task="multiclass").to(device)
        self.f1 = F1Score(num_classes=num_labels, task="multiclass").to(device)
        self.mcc = MatthewsCorrCoef(num_classes=num_labels, task="multiclass").to(device)
        self.top2_acc = Accuracy(top_k = 2, num_classes=num_labels, task="multiclass").to(device)

    def split_features(self, combined_feature):
        # Split the combined features into MFCC and spectrogram features
        x_mfcc = combined_feature[:, :self.mfcc_dim, :]
        x_spectrogram = combined_feature[:, self.mfcc_dim:, :]
        return x_mfcc, x_spectrogram

    def train_step(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        train_iterator = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for batch in train_iterator:
            combined_feature, labels = batch
            x_mfcc, x_spectrogram = self.split_features(combined_feature)
            inputs_mfcc, inputs_spectrogram, labels = x_mfcc.to(self.device), x_spectrogram.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs_mfcc, inputs_spectrogram)
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
                # Split the combined features into MFCC and spectrogram features
                combined_feature, labels = batch
                x_mfcc, x_spectrogram = self.split_features(combined_feature)
                inputs_mfcc, inputs_spectrogram, labels = x_mfcc.to(self.device), x_spectrogram.to(self.device), labels.to(self.device)
                
                # Pass both inputs to the model
                outputs = self.model(inputs_mfcc, inputs_spectrogram)
                loss = self.loss_criterion(outputs, labels)

                total_loss += loss.item()

                # Update metrics
                self.accuracy(outputs, labels)
                self.precision(outputs, labels)
                self.recall(outputs, labels)
                self.f1(outputs, labels)
                self.mcc(outputs, labels)
                self.top2_acc(outputs,labels)

                iterator.set_postfix(loss=total_loss / (iterator.n + 1))

        iterator.close()

        # Calculate final metrics if phase is "Testing"
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

    


class CombinedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, mfcc_directory, spectrogram_directory, mfcc_augmented_directory=None, spectrogram_augmented_directory=None):
        self.mfcc_files = self._get_files(mfcc_directory)
        self.spectrogram_files = self._get_files(spectrogram_directory)

        if mfcc_augmented_directory:
            self.mfcc_files += self._get_files(mfcc_augmented_directory)
        if spectrogram_augmented_directory:
            self.spectrogram_files += self._get_files(spectrogram_augmented_directory)

        assert len(self.mfcc_files) == len(self.spectrogram_files), "MFCC and Spectrogram file counts do not match"

        self.mfcc_data = []
        self.spectrogram_data = []
        self.labels = []

        for mfcc_file, spectrogram_file in zip(self.mfcc_files, self.spectrogram_files):
            mfcc_features, mfcc_labels = torch.load(mfcc_file)
            spectrogram_features, _ = torch.load(spectrogram_file)

            self.mfcc_data.extend(mfcc_features)
            self.spectrogram_data.extend(spectrogram_features)
            self.labels.extend(mfcc_labels)

    def _get_files(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mfcc_data[idx], self.spectrogram_data[idx], self.labels[idx]






def create_datasets(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


combined_dataset = CombinedAudioDataset(
mfcc_directory = r"E:\speech_datasets\mfcc_feature_extracted_dataset",
spectrogram_directory = r"E:\speech_datasets\spectrogram_feature_extracted_dataset",
mfcc_augmented_directory = r"E:\speech_datasets\mfcc_feature_extracted_dataset_augmented", # Optional
spectrogram_augmented_directory = r"E:\speech_datasets\spectrogram_feature_extracted_dataset_augmented" # Optional 
)


# Split the dataset
train_dataset, val_dataset, test_dataset = create_datasets(combined_dataset)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mfcc_dim = 250  # Set the correct MFCC dimension
spectrogram_dim = 251  # Set the correct Spectrogram dimension

model = SpeechTransformer.CustomTransformerClassifier(mfcc_dim=mfcc_dim, spectrogram_dim=spectrogram_dim, num_classes=6, num_heads=16, dim_feedforward=2048, num_layers=4, dropout=0.1)
# Create the Classifier instance
classifier = Classifier(model, device, num_labels=6, log_dir="logs", mfcc_dim=mfcc_dim, spectrogram_dim=spectrogram_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)

# Training loop
num_epochs = 13
for epoch in range(num_epochs):
    classifier.train_step(train_loader, optimizer, epoch)
    classifier.val_test_step(val_loader, "Validation")

# Testing
test_results = classifier.val_test_step(test_loader, "Testing")
print("\nTest Metrics:")
print(f"Accuracy: {test_results['accuracy']:.4f}")
print(f"Precision: {test_results['precision']:.4f}")
print(f"Recall: {test_results['recall']:.4f}")
print(f"F1 Score: {test_results['f1']:.4f}")
print(f"Matthews Correlation Coefficient: {test_results['mcc']:.4f}")
print(f"Top 2 Accuracy: {test_results['top 2 accuracy']:.4f}")
