import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
import numpy as np
import os
import soundfile as sf






def extract_mfccs(waveform, sample_rate, n_mfcc=30, win_length=400, hop_length=160, n_mels=30, n_fft=1024):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'win_length': win_length, 
            'hop_length': hop_length, 
            'n_mels': n_mels,
            'n_fft': n_fft
        }
    )
    mfcc = mfcc_transform(waveform)
    return ensure_3d_tensor(mfcc)

def ensure_3d_tensor(tensor):
    if tensor.dim() == 2:  # [features, time]
        tensor = tensor.unsqueeze(0)  # Add channel dimension: [1, features, time]
    return tensor

def extract_spectrogram(waveform, sample_rate, n_mels=30, win_length=400, hop_length=160, n_fft=1024):
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft
    )
    spectrogram = spectrogram_transform(waveform)
    return ensure_3d_tensor(spectrogram)




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





