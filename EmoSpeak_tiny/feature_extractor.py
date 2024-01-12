import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

import os

# Feature Extraction Function
def extract_mfccs(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)
    mfcc = mfcc_transform(waveform)
    return mfcc

def extract_spectrogram(audio_path, n_mels=48):
    waveform, sample_rate = torchaudio.load(audio_path)
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    spectrogram = spectrogram_transform(waveform)
    return spectrogram
