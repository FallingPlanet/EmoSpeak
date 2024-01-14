import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import feature_extractor 
from feature_extractor import apply_augmentations, process_and_augment_directory, create_directory_if_not_exists

import os

import soundfile as sf

def save_waveform(waveform, sample_rate, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Save the waveform
    sf.write(file_path, waveform.t().numpy(), sample_rate)


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

# Dataset processing functions for CREMA, RAVDESS, TESS, SAVEE
def process_dataset(directory, dataset_mapping, unified_mapping, save_directory, augmented_directory, num_augmentations_per_file, feature_extractor_function):
    data = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root, filename)
                label = extract_label_from_dataset(filename, dataset_mapping, unified_mapping)
                if label != -1:
                    # Process audio file with the given feature extraction function
                    process_audio_file(full_path, filename, label, save_directory, augmented_directory, num_augmentations_per_file, data, feature_extractor_function)
    return data

def extract_label_from_crema(filename, crema_mapping, unified_mapping):
    emotion_identifier = filename.split('_')[2]
    emotion = crema_mapping.get(emotion_identifier)
    return unified_mapping.get(emotion, -1)  # Returns -1 if the emotion is not in unified mapping

def extract_label_from_ravdess(filename, ravdess_mapping, unified_mapping):
    components = filename.split('-')
    if len(components) >= 3:
        emotion_code = components[2]
        emotion = ravdess_mapping.get(emotion_code)
        return unified_mapping.get(emotion, -1)
    else:
        return -1

def extract_label_from_tess(filename, tess_mapping, unified_mapping):
    emotion = filename.split('_')[2].lower()
    emotion = tess_mapping.get(emotion)
    return unified_mapping.get(emotion, -1)

def extract_label_from_savee(filename, savee_mapping, unified_mapping):
    emotion = filename[:2].lower() if filename[0] == 's' else filename[0].lower()
    emotion = savee_mapping.get(emotion)
    return unified_mapping.get(emotion, -1)

def extract_label_from_emo_db(filename, emo_db_mapping, unified_mapping):
    emotion_identifier = filename[2]
    emotion = emo_db_mapping.get(emotion_identifier)
    return unified_mapping.get(emotion, -1)

def extract_label_from_emovo(filename, emovo_mapping, unified_mapping):
    emotion_identifier = filename.split('_')[0].lower()
    emotion = emovo_mapping.get(emotion_identifier)
    return unified_mapping.get(emotion, -1)

def extract_label_from_dataset(filename, dataset_mapping, unified_mapping):
    if dataset_mapping == crema_mapping:
        return extract_label_from_crema(filename, dataset_mapping, unified_mapping)
    elif dataset_mapping == ravdess_mapping:
        return extract_label_from_ravdess(filename, dataset_mapping, unified_mapping)
    elif dataset_mapping == tess_mapping:
        return extract_label_from_tess(filename, dataset_mapping, unified_mapping)
    elif dataset_mapping == savee_mapping:
        return extract_label_from_savee(filename, dataset_mapping, unified_mapping)
    elif dataset_mapping == emo_db_mapping:
        return extract_label_from_emo_db(filename, dataset_mapping, unified_mapping)
    elif dataset_mapping == emovo_mapping:
        return extract_label_from_emovo(filename, dataset_mapping, unified_mapping)
    else:
        return -1



def process_audio_file(full_path, filename, label, save_directory, augmented_directory, num_augmentations_per_file, data, feature_extractor_function):
    waveform, sample_rate = torchaudio.load(full_path)
    # Process original sample
    original_feature = feature_extractor_function(waveform, sample_rate)
    
 
    save_features_and_labels([original_feature], [label], os.path.splitext(filename)[0], save_directory)
    data[full_path] = label

    # Process augmented samples
    augment_audio(filename, waveform, sample_rate, label, augmented_directory, num_augmentations_per_file, data, feature_extractor_function)

def augment_audio(filename, waveform, sample_rate, label, augmented_directory, num_augmentations_per_file, data, feature_extractor_function):
    for i in range(num_augmentations_per_file):
        augmented_waveform = apply_augmentations(waveform, sample_rate)
        augmented_feature = feature_extractor_function(augmented_waveform, sample_rate)  # Pass both waveform and sample_rate
        augmented_filename = f"{os.path.splitext(filename)[0]}_augmented_{i}"
        save_features_and_labels([augmented_feature], [label], augmented_filename, augmented_directory)


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

def process_and_save_dataset(directory, mapping, unified_mapping, feature_extractor_function, save_directory, augmented_directory, num_augmentations_per_file):
    data = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root, filename)
                label = mapping.get(filename, unified_mapping)
                if label != -1:
                    # Process and save original sample
                    waveform, sample_rate = torchaudio.load(full_path)
                    feature = feature_extractor_function(waveform)
                    save_features_and_labels([feature], [label], os.path.splitext(filename)[0], save_directory)

                    # Process and save augmented samples
                    for i in range(num_augmentations_per_file):
                        augmented_waveform = apply_augmentations(waveform, sample_rate)
                        augmented_feature = feature_extractor_function(augmented_waveform)
                        augmented_filename = f"{os.path.splitext(filename)[0]}_augmented_{i}"
                        save_features_and_labels([augmented_feature], [label], augmented_filename, augmented_directory)
                        data[os.path.join(augmented_directory, augmented_filename + '.pt')] = label

    return data

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
    "Boredom": 0,
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
emo_db_directory_path = r"E:\speech_datasets\Emo-DB_dataset\wav"
emo_db_mapping = {
    "W": "Angry",    # Ärger (Wut)
    "L": "Neutral",  # Langeweile (mapped to Neutral)
    "E": "Disgust",  # Ekel
    "A": "Fearful",  # Angst
    "F": "Happy",    # Freude
    "T": "Sad",      # Trauer
    # Note: There is no direct mapping for "Surprise" in Emo-DB
}
emovo_directory_path = r"E:\speech_datasets\EMOVO_dataset"
emovo_mapping = {
    "dis": "Disgust",
    "pau": "Fearful",
    "rab": "Angry",
    "gio": "Happy",
    "sor": "Surprise",
    "tri": "Sad",
    "neu": "Neutral"
}


def merge_datasets(*datasets):
    merged_data = {}
    for dataset in datasets:
        merged_data.update(dataset)
    return merged_data

# Define separate directories for saving MFCCs and Spectrograms
save_directory_mfcc = r"E:\speech_datasets\mfcc_feature_extracted_dataset"
augment_save_directory_mfcc = r"E:\speech_datasets\mfcc_feature_extracted_dataset_augmented"
save_directory_spectrogram = r"E:\speech_datasets\spectrogram_feature_extracted_dataset"
augment_save_directory_spectrogram = r"E:\speech_datasets\spectrogram_feature_extracted_dataset_augmented"

directories_to_create = [
    save_directory_mfcc, augment_save_directory_mfcc, 
    save_directory_spectrogram, augment_save_directory_spectrogram
]

# Create directories if they don't exist
for directory in directories_to_create:
    create_directory_if_not_exists(directory)
# Define the feature extraction functions
def extract_mfccs(waveform, sample_rate):
    return feature_extractor.extract_mfccs(waveform, sample_rate)

def extract_spectrogram(waveform, sample_rate):
    return feature_extractor.extract_spectrogram(waveform, sample_rate)



# Process and save each dataset for MFCCs and Spectrograms
for dataset_path, mapping in [(crema_directory_path, crema_mapping), (ravdess_directory_path, ravdess_mapping), (tess_directory_path, tess_mapping), (savee_directory_path, savee_mapping), (emo_db_directory_path, emo_db_mapping), (emovo_directory_path, emovo_mapping)]:
    # Process for MFCCs
    mfcc_data = process_dataset(dataset_path, mapping, unified_mapping, save_directory_mfcc, augment_save_directory_mfcc, 4, extract_mfccs)
    
    # Process for Spectrograms
    spectrogram_data = process_dataset(dataset_path, mapping, unified_mapping, save_directory_spectrogram, augment_save_directory_spectrogram, 4, extract_spectrogram)

    # Save MFCC data
    mfccs, mfcc_labels = process_datasets_for_features(mfcc_data, extract_mfccs)
    save_features_and_labels(mfccs, mfcc_labels, 'mfcc', save_directory_mfcc)

    # Save Spectrogram data
    spectrograms, spectrogram_labels = process_datasets_for_features(spectrogram_data, extract_spectrogram)

