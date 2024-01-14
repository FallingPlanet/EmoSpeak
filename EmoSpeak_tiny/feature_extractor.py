import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
import numpy as np
import os
import soundfile as sf






def extract_mfccs(waveform, sample_rate, n_mfcc=40, win_length=400, hop_length=160, n_mels=128):
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                                n_mfcc=n_mfcc,
                                                melkwargs={'win_length': win_length, 
                                                           'hop_length': hop_length, 
                                                           'n_mels': n_mels})
    mfcc = mfcc_transform(waveform)
    return mfcc

def extract_spectrogram(waveform, sample_rate, n_mels=128, win_length=400, hop_length=160):
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                 n_mels=n_mels,
                                                                 win_length=win_length,
                                                                 hop_length=hop_length)
    spectrogram = spectrogram_transform(waveform)
    return spectrogram



def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_waveform(waveform, sample_rate, file_path):
    create_directory_if_not_exists(os.path.dirname(file_path))
    sf.write(file_path, waveform.t().numpy(), sample_rate)

def apply_augmentations(waveform, sample_rate):
    # Add Background Noise
    noise = torch.randn(waveform.size())
    noise_level = random.uniform(0.001, 0.1)
    waveform_with_noise = waveform + noise_level * noise

    # Dynamic Range Compression (Volume Adjust)
    volume_factor = random.uniform(0.5, 1.5)
    vol_transform = torchaudio.transforms.Vol(volume_factor)
    waveform_volume_adjusted = vol_transform(waveform_with_noise)

    return waveform_volume_adjusted

# Function to process all audio files within a directory and save augmented samples
def process_and_augment_directory(directory, save_directory, augmented_directory, num_augmentations_per_file=3):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.wav'):
            full_path = os.path.join(directory, filename)
            waveform, sample_rate = torchaudio.load(full_path)

            # Save original sample
            original_save_path = os.path.join(save_directory, filename)
            save_waveform(waveform, sample_rate, original_save_path)

            # Generate and save augmented samples
            for i in range(num_augmentations_per_file):
                augmented_waveform = apply_augmentations(waveform, sample_rate)
                augmented_filename = f"{os.path.splitext(filename)[0]}_augmented_{i}.wav"




