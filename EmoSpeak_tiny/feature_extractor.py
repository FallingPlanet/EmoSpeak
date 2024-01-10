import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Feature Extraction Function
def extract_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)
    mfcc = mfcc_transform(waveform)
    return mfcc

# Custom Dataset Class
class SERDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        mfcc = extract_features(self.audio_paths[idx])
        label = self.labels[idx]
        return mfcc, label

def extract_label_from_crema(filename, crema_mapping, unified_mapping):
    emotion_identifier = filename.split('_')[2]
    emotion = crema_mapping.get(emotion_identifier)
    return unified_mapping.get(emotion, -1)  # Returns -1 if the emotion is not in unified mapping

def process_crema(directory, mapping):
    labels = {}
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            label = extract_label_from_crema(filename, mapping)
            labels[filename] = label
    return labels

def process_ravdess(directory, mapping):
    labels = {
        
    }
    for root, dirs, files in os.walk(directory):
        for filename  in files:
            if filename.endswith('.wav'):
                full_path = os.path.join(root, filename)
                label = extract_label_from_ravdess(filename, mapping)
                labels[full_path] = label

def extract_label_from_ravdess(filename, mapping):
    components = filename.split('-')
    if len(components) >= 3:
        emotion_code = components[2]
        return mapping.get(emotion_code, -1)
    else:
        return -1
    
unified_mapping = {
    "Neutral": 0,
    "Calm": 1,
    "Happy": 2,
    "Sad": 3,
    "Angry": 4,
    "Fearful": 5,
    "Disgust": 6,
    "Surprised": 7
    # ...
}
crema_directory_path = r"E:\speech_datasets\CREMA-D_dataset\CREMA-D-master\AudioWAV"
crema_mapping = {
    "NEU": "Neutral",
    "HAP": "Happy",
    "SAD": "Sad",
    "ANG": "Angry",
    "FEA": "Fearful",
    "DIS": "Disgust"
    # Note: "Calm" and "Surprised" are not present in CREMA-D
}
ravdess_directory_path = ""
ravdess_mapping = {
    "01": 0,  # Neutral
    "02": 1,  # Calm
    "03": 2,  # Happy
    "04": 3,  # Sad
    "05": 4,  # Angry
    "06": 5,  # Fearful
    "07": 6,  # Disgust
    "08": 7   # Surprised
}
tess_mapping = {
    "neutral": 0,  # Neutral
    "calm": 1,  # Calm
    "": 2,  # Happy
    "04": 3,  # Sad
    "05": 4,  # Angry
    "06": 5,  # Fearful
    "07": 6,  # Disgust
    "08": 7   # Surprised
}
