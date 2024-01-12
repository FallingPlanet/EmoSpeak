import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import model as SpeechTransformer
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
from FallingPlanet.orbit.utils.Metrics import TinyEmoBoard

# Assuming SpeechTransformer model is defined
class Classifier:
    def __init__(self, model, device, num_labels, log_dir):
        self.model = model.to(device)
        self.device = device
        self.loss_criterion = nn.CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)

        self.accuracy = Accuracy(num_classes=num_labels).to(device)
        self.precision = Precision(num_classes=num_labels).to(device)
        self.recall = Recall(num_classes=num_labels).to(device)
        self.f1 = F1Score(num_classes=num_labels).to(device)
        self.mcc = MatthewsCorrCoef(num_classes=num_labels).to(device)

    def train_step(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        train_iterator = tqdm(dataloader, desc=f"Training Epoch {epoch}")

        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_iterator.set_postfix(loss=total_loss / (train_iterator.n + 1))

        train_iterator.close()

    def val_test_step(self, dataloader, phase="Validation"):
        self.model.eval()
        total_loss = 0.0
        iterator = tqdm(dataloader, desc=f"{phase} Epoch")

        with torch.no_grad():
            for inputs, labels in iterator:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, labels)

                total_loss += loss.item()

                # Update metrics
                self.accuracy(outputs, labels)
                self.precision(outputs, labels)
                self.recall(outputs, labels)
                self.f1(outputs, labels)
                self.mcc(outputs, labels)

                iterator.set_postfix(loss=total_loss / (iterator.n + 1))

        iterator.close()

        # Log metrics
        self.writer.log_scalar(f'{phase}/Loss', total_loss / len(dataloader))
        self.writer.log_scalar(f'{phase}/Accuracy', self.accuracy.compute())
        self.writer.log_scalar(f'{phase}/Precision', self.precision.compute())
        self.writer.log_scalar(f'{phase}/Recall', self.recall.compute())
        self.writer.log_scalar(f'{phase}/F1', self.f1.compute())
        self.writer.log_scalar(f'{phase}/MCC', self.mcc.compute())
    

# Load features and labels
mfcc_features = torch.load("E:/speech_datasets/feature_extracted_dataset/mfcc_features.pt")
mfcc_labels = torch.load("E:/speech_datasets/feature_extracted_dataset/mfcc_labels.pt")

# Splitting datasets
dataset_size = len(mfcc_features)
train_size = int(0.7 * dataset_size)
val_test_size = dataset_size - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size

train_dataset, val_test_dataset = random_split(range(dataset_size), [train_size, val_test_size])
val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechTransformer.CustomTransformerClassifier(input_dim=40, num_heads=8, num_layers=4, num_classes=10).to(device)
classifier = Classifier(model, device, num_labels=10, log_dir="path/to/log/dir")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    classifier.train_step(train_loader, optimizer, epoch)
    classifier.val_test_step(val_loader, "Validation")

# Testing
test_results = classifier.val_test_step(test_loader, "Testing")

