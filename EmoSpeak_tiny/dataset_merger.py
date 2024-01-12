import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import feature_extractor 
import os


# Custom Dataset Class
class SERDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        transformation = feature_extractor.extract_mfccs(self.audio_paths[idx])
        label = self.labels[idx]
        return transformation, label

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

def process_datasets_for_features(dataset, feature_extractor_function):
    all_features = []
    all_labels = []
    for audio_path, label in dataset.items():
        feature = feature_extractor_function(audio_path)
        all_features.append(feature)
        all_labels.append(label)
    return all_features, all_labels

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # Get the maximum length in the time dimension
    max_len = max([s.size(-1) for s in sequences])

    # Pad all sequences to this length
    padded_sequences = []
    for s in sequences:
        if s.size(-1) < max_len:
            # Padding size for the last dimension
            padding_size = (0, max_len - s.size(-1))
            padded_s = torch.nn.functional.pad(s, padding_size, 'constant', padding_value)
            padded_sequences.append(padded_s)
        else:
            padded_sequences.append(s)

    return torch.stack(padded_sequences, dim=0 if batch_first else 1)

# Usage in your save_features_and_labels function
def save_features_and_labels(features, labels, feature_type, save_path):
    # Pad features before stacking
    features_tensor = pad_sequence(features, batch_first=True)  # batch_first depends on your model's requirement
    labels_tensor = torch.tensor(labels)

    torch.save(features_tensor, os.path.join(save_path, f'{feature_type}_features.pt'))
    torch.save(labels_tensor, os.path.join(save_path, f'{feature_type}_labels.pt'))

    
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
crema_directory_path = r"E:\speech_datasets\CREMA-D_dataset\AudioWAV"
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


def merge_datasets(*datasets):
    merged_data = {}
    for dataset in datasets:
        merged_data.update(dataset)
    return merged_data

# Process and merge each dataset
crema_data = process_crema(crema_directory_path, crema_mapping, unified_mapping)
ravdess_data = process_ravdess(ravdess_directory_path, ravdess_mapping, unified_mapping)
tess_data = process_tess(tess_directory_path, tess_mapping, unified_mapping)
savee_data = process_savee(savee_directory_path, savee_mapping, unified_mapping)

# Combine data from all datasets
all_data = merge_datasets(crema_data, ravdess_data, tess_data, savee_data)

# Process for MFCCs and Spectrograms
mfccs, labels = process_datasets_for_features(all_data, feature_extractor.extract_mfccs)
spectrograms, _ = process_datasets_for_features(all_data, feature_extractor.extract_spectrogram)

# Save the data to files
save_directory = r"E:\speech_datasets\feature_extracted_dataset"
save_features_and_labels(mfccs, labels, 'mfcc', save_directory)
save_features_and_labels(spectrograms, labels, 'spectrogram', save_directory)