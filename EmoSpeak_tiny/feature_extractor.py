import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
import numpy as np
import os
import soundfile as sf


def apply_augmentations(waveform, sample_rate):
    # Add Background Noise
    noise = torch.randn(waveform.size())
    noise_level = random.uniform(0.001, 0.1)  # Adjust the noise level
    waveform_with_noise = waveform + noise_level * noise

    # Dynamic Range Compression (Volume Adjust)
    volume_factor = random.uniform(0.5, 1.5)  # Adjust the volume factor
    vol_transform = torchaudio.transforms.Vol(volume_factor)
    waveform_volume_adjusted = vol_transform(waveform_with_noise)

    return waveform_volume_adjusted

def save_augmented_sample(waveform, sample_rate, save_path):
    sf.write(save_path, waveform.t().numpy(), sample_rate)

def create_and_save_augmented_samples(audio_paths, save_directory, num_augmentations_per_file=3):
    for path in audio_paths:
        waveform, sample_rate = torchaudio.load(path)
        base_filename = os.path.basename(path)

        # Save original sample
        original_save_path = os.path.join(save_directory, base_filename)
        save_augmented_sample(waveform, sample_rate, original_save_path)

        # Generate and save augmented samples
        for i in range(num_augmentations_per_file):
            augmented_waveform = apply_augmentations(waveform, sample_rate)
            augmented_filename = f"{os.path.splitext(base_filename)[0]}_augmented_{i}.wav"
            augmented_save_path = os.path.join(save_directory, augmented_filename)
            save_augmented_sample(augmented_waveform, sample_rate, augmented_save_path)
