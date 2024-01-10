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

def process_crema(directory, crema_mapping, unified_mapping):
    data = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            full_path = os.path.join(directory, filename)
            label = extract_label_from_crema(filename, crema_mapping, unified_mapping)
            if label != -1:
                data[full_path] = label
    return data

def extract_label_from_ravdess(filename, ravdess_mapping, unified_mapping):
    components = filename.split('-')
    if len(components) >= 3:
        emotion_code = components[2]
        emotion = ravdess_mapping.get(emotion_code)
        return unified_mapping.get(emotion, -1)
    else:
        return -1

def process_ravdess(directory, ravdess_mapping, unified_mapping):
    data = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root, filename)
                label = extract_label_from_ravdess(filename, ravdess_mapping, unified_mapping)
                if label != -1:
                    data[full_path] = label
    return data

def process_tess(directory, tess_mapping, unified_mapping):
    data = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root, filename)
                emotion = filename.split('_')[2].lower()
                emotion = tess_mapping.get(emotion)
                label = unified_mapping.get(emotion, -1)
                if label != -1:
                    data[full_path] = label
    return data

def process_savee(directory, savee_mapping, unified_mapping):
    data = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root, filename)
                emotion = filename[:2].lower() if filename[0] == 's' else filename[0].lower()
                emotion = savee_mapping.get(emotion)
                label = unified_mapping.get(emotion, -1)
                if label != -1:
                    data[full_path] = label
    return data

unified_mapping = {
    "Neutral": 0,
    "Calm": 0,
    "Happy": 1,
    "Sad": 2,
    "Angry": 3,
    "Fearful": 4,
    "Disgust": 5,
    "Surprise": 6
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
ravdess_directory_path = r"E:\speech_datasets\Ravdess_dataset\audio_speech_actors_01-24"
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
tess_directory_path = r"E:\speech_datasets\TESS_dataset\TESS Toronto emotional speech set data\TESS Toronto emotional speech set data"
tess_mapping = {
    "neutral": 0,  # Neutral
    "happy": 2,  # Happy
    "fear": 3,  # Sad
    "anrgy": 4,  # Angry
    "fear": 5,  # Fearful
    "disgust": 6,  # Disgust
    "pleasent_surprise": 7   # Surprised
}
savee_directory_path = r"E:\speech_datasets\SAVEE_dataset\AudioData\AudioData"
savee_mapping = {
    "a": 4,  # Angry
    "d": 6,  # Disgust
    "sa": 2,  # Sad
    "su": 7,  # Surprise
    "n": 0,  # Neutral (same as Calm in your unified mapping)
    "h": 1,  # Happy
    "f": 5   # Fear
}

# Process each dataset
crema_data = process_crema(crema_directory_path, crema_mapping, unified_mapping)
ravdess_data = process_ravdess(ravdess_directory_path, ravdess_mapping, unified_mapping)
tess_data = process_tess(tess_directory_path, tess_mapping, unified_mapping)
savee_data = process_savee(savee_directory_path, savee_mapping, unified_mapping)

# Combine and filter the data
all_data = {**crema_data, **ravdess_data, **tess_data, **savee_data}  # Merge dictionaries

# Separate paths and labels
all_audio_paths = list(all_data.keys())
all_labels = list(all_data.values())

# Create dataset and dataloader
ser_dataset = SERDataset(all_audio_paths, all_labels)
ser_dataloader = DataLoader(ser_dataset, batch_size=32, shuffle=True)