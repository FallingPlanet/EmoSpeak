import os
import torchaudio
import torch
from torch.utils.data import Dataset
import feature_extractor

class SERDataset(Dataset):
    def __init__(self, directory, mapping, unified_mapping, feature_extraction_func_name='extract_mfccs', is_train=False, num_augmentations=0):
        self.directory = directory
        self.mapping = mapping
        self.unified_mapping = unified_mapping
        self.is_train = is_train
        self.num_augmentations = num_augmentations
        self.feature_extraction_function = getattr(feature_extractor, feature_extraction_func_name)
        self.data = self.load_data()
        self.max_length = 0  # Initialize max_length

    def load_data(self):
        data = []
        max_length = 0

        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.lower().endswith('.wav'):
                    full_path = os.path.join(root, filename)
                    waveform, sample_rate = torchaudio.load(full_path)
                    feature = self.feature_extraction_function(waveform, sample_rate)

                    max_length = max(max_length, feature.shape[-1])

                    label = self.extract_label_from_filename(root, filename)
                    if label != -1:
                        data.append((feature, label))

                        if self.is_train:
                            for _ in range(self.num_augmentations):
                                augmented_waveform = feature_extractor.apply_augmentations(waveform, sample_rate)
                                augmented_feature = self.feature_extraction_function(augmented_waveform, sample_rate)
                                data.append((augmented_feature, label))

        self.max_length = max_length
        return data

    def extract_label_from_filename(self, root, filename):
        if 'CREMA-D' in root:
            emotion_identifier = filename.split('_')[2]
            return self.unified_mapping.get(self.mapping.get(emotion_identifier, None), -1)
        elif 'Emo-DB' in root:
            emotion_identifier = filename[5]
            return self.unified_mapping.get(self.mapping.get(emotion_identifier, None), -1)
        elif 'EMOVO' in root:
            emotion_identifier = filename.split('-')[0]
            return self.unified_mapping.get(self.mapping.get(emotion_identifier, None), -1)
        elif 'Ravdess' in root:
            emotion_identifier = filename.split('-')[2]
            return self.unified_mapping.get(self.mapping.get(emotion_identifier, None), -1)
        elif 'SAVEE' in root:
            if filename.startswith('n'):
                emotion_identifier = 'n'
            else:
                emotion_identifier = filename[0:2]
            return self.unified_mapping.get(self.mapping.get(emotion_identifier, None), -1)
        elif 'TESS' in root:
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_identifier = parts[2].lower().split('.')[0]
            return self.unified_mapping.get(self.mapping.get(emotion_identifier, None), -1)
        return -1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    


def merge_and_save_datasets(datasets, save_path):
    merged_features = []
    merged_labels = []

    for dataset in datasets:
        for feature, label in dataset:
            merged_features.append(feature)
            merged_labels.append(label)

    # Save the combined dataset as a list of tensors
    data_dict = {'features': merged_features, 'labels': merged_labels}
    torch.save(data_dict, save_path)




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

# CREMA-D
crema_dataset_train = SERDataset(
    directory=crema_directory_path, 
    mapping=crema_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=True, 
    num_augmentations=4
)

# RAVDESS
ravdess_dataset_train = SERDataset(
    directory=ravdess_directory_path, 
    mapping=ravdess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=True, 
    num_augmentations=4
)

# TESS
tess_dataset_train = SERDataset(
    directory=tess_directory_path, 
    mapping=tess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=True, 
    num_augmentations=4
)

# SAVEE
savee_dataset_train = SERDataset(
    directory=savee_directory_path, 
    mapping=savee_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=True, 
    num_augmentations=4
)

# Emo-DB
emo_db_dataset_train = SERDataset(
    directory=emo_db_directory_path, 
    mapping=emo_db_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=True, 
    num_augmentations=4
)

# EMOVO
emovo_dataset_train = SERDataset(
    directory=emovo_directory_path, 
    mapping=emovo_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=True, 
    num_augmentations=4
)


# CREMA-D
crema_dataset_val = SERDataset(
    directory=crema_directory_path, 
    mapping=crema_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# RAVDESS
ravdess_dataset_val = SERDataset(
    directory=ravdess_directory_path, 
    mapping=ravdess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# TESS
tess_dataset_val = SERDataset(
    directory=tess_directory_path, 
    mapping=tess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# SAVEE
savee_dataset_val = SERDataset(
    directory=savee_directory_path, 
    mapping=savee_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# Emo-DB
emo_db_dataset_val = SERDataset(
    directory=emo_db_directory_path, 
    mapping=emo_db_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# EMOVO
emovo_dataset_val = SERDataset(
    directory=emovo_directory_path, 
    mapping=emovo_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)



# CREMA-D
crema_dataset_test = SERDataset(
    directory=crema_directory_path, 
    mapping=crema_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# RAVDESS
ravdess_dataset_test = SERDataset(
    directory=ravdess_directory_path, 
    mapping=ravdess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# TESS
tess_dataset_test = SERDataset(
    directory=tess_directory_path, 
    mapping=tess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# SAVEE
savee_dataset_test = SERDataset(
    directory=savee_directory_path, 
    mapping=savee_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# Emo-DB
emo_db_dataset_test = SERDataset(
    directory=emo_db_directory_path, 
    mapping=emo_db_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# EMOVO
emovo_dataset_test = SERDataset(
    directory=emovo_directory_path, 
    mapping=emovo_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_mfccs', 
    is_train=False, 
    num_augmentations=4
)

# CREMA-D
crema_dataset_train_spectrogram = SERDataset(
    directory=crema_directory_path, 
    mapping=crema_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=True, 
    num_augmentations=4
)

# RAVDESS
ravdess_dataset_train_spectrogram = SERDataset(
    directory=ravdess_directory_path, 
    mapping=ravdess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=True, 
    num_augmentations=4
)

# TESS
tess_dataset_train_spectrogram = SERDataset(
    directory=tess_directory_path, 
    mapping=tess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=True, 
    num_augmentations=4
)

# SAVEE
savee_dataset_train_spectrogram = SERDataset(
    directory=savee_directory_path, 
    mapping=savee_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=True, 
    num_augmentations=4
)

# Emo-DB
emo_db_dataset_train_spectrogram = SERDataset(
    directory=emo_db_directory_path, 
    mapping=emo_db_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=True, 
    num_augmentations=4
)

# EMOVO
emovo_dataset_train_spectrogram = SERDataset(
    directory=emovo_directory_path, 
    mapping=emovo_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=True, 
    num_augmentations=4
)

crema_dataset_val_spectrogram = SERDataset(
    directory=crema_directory_path, 
    mapping=crema_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

ravdess_dataset_val_spectrogram = SERDataset(
    directory=ravdess_directory_path, 
    mapping=ravdess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

tess_dataset_val_spectrogram = SERDataset(
    directory=tess_directory_path, 
    mapping=tess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

savee_dataset_val_spectrogram = SERDataset(
    directory=savee_directory_path, 
    mapping=savee_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

emo_db_dataset_val_spectrogram = SERDataset(
    directory=emo_db_directory_path, 
    mapping=emo_db_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

emovo_dataset_val_spectrogram = SERDataset(
    directory=emovo_directory_path, 
    mapping=emovo_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

crema_dataset_test_spectrogram = SERDataset(
    directory=crema_directory_path, 
    mapping=crema_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

ravdess_dataset_test_spectrogram = SERDataset(
    directory=ravdess_directory_path, 
    mapping=ravdess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

tess_dataset_test_spectrogram = SERDataset(
    directory=tess_directory_path, 
    mapping=tess_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

savee_dataset_test_spectrogram = SERDataset(
    directory=savee_directory_path, 
    mapping=savee_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

emo_db_dataset_test_spectrogram = SERDataset(
    directory=emo_db_directory_path, 
    mapping=emo_db_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)

emovo_dataset_test_spectrogram = SERDataset(
    directory=emovo_directory_path, 
    mapping=emovo_mapping, 
    unified_mapping=unified_mapping, 
    feature_extraction_func_name='extract_spectrogram', 
    is_train=False
)




# Example usage:

# MFCC datasets from different sources
mfcc_datasets = [crema_dataset_train, ravdess_dataset_train, tess_dataset_train, savee_dataset_train, emo_db_dataset_train, emovo_dataset_train]
merge_and_save_datasets(mfcc_datasets, r'E:\speech_datasets\FP_EmoSpeak_MFCCs_Train.pt')

# Spectrogram datasets from different sources
spectrogram_datasets = [crema_dataset_train_spectrogram, ravdess_dataset_train_spectrogram, tess_dataset_train_spectrogram, savee_dataset_train_spectrogram, emo_db_dataset_train_spectrogram, emovo_dataset_train_spectrogram]
merge_and_save_datasets(spectrogram_datasets, r'E:\speech_datasets\FP_EmoSpeak_Spectrograms_Train.pt')

# Repeat for validation and test datasets
# For Validation
merge_and_save_datasets([crema_dataset_val, ravdess_dataset_val, tess_dataset_val, savee_dataset_val, emo_db_dataset_val, emovo_dataset_val], r'E:\speech_datasets\FP_EmoSpeak_MFCCs_Val.pt')
merge_and_save_datasets([crema_dataset_val_spectrogram, ravdess_dataset_val_spectrogram, tess_dataset_val_spectrogram, savee_dataset_val_spectrogram, emo_db_dataset_val_spectrogram, emovo_dataset_val_spectrogram], r'E:\speech_datasets\FP_EmoSpeak_Spectrograms_Val.pt')

# For Test
merge_and_save_datasets([crema_dataset_test, ravdess_dataset_test, tess_dataset_test, savee_dataset_test, emo_db_dataset_test, emovo_dataset_test], r'E:\speech_datasets\FP_EmoSpeak_MFCCs_Test.pt')
merge_and_save_datasets([crema_dataset_test_spectrogram, ravdess_dataset_test_spectrogram, tess_dataset_test_spectrogram, savee_dataset_test_spectrogram, emo_db_dataset_test_spectrogram, emovo_dataset_test_spectrogram], r'E:\speech_datasets\FP_EmoSpeak_Spectrograms_Test.pt')
