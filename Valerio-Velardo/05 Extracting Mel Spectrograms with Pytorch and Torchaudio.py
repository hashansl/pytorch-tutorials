import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio



class UrbanSoundDataset(Dataset):

    def __init__(self,annotation_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        # len(usd)
        return len(self.annotations)

    def __getitem__(self, index):
        # a_list[1] --> a_list.__getitem__(1) <----Under the hood
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        #signal -> (num_channels, samples) ---> (2, 16000) ---> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self.transformation(signal)

        return signal,label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resample(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:  #(2, 1000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])

        return path

    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index, 6]

if __name__ == "__main__":

    ANNOTATION_FILE = "/media/hashan/Laptop Hard Bk/python projects-B/Research/Sound/jupyter/Audio_classification_iron/New Data/Urban_sound_8k/UrbanSound8K.csv"
    AUDIO_DIR = "/media/hashan/Laptop Hard Bk/python projects-B/Research/Sound/jupyter/Audio_classification_iron/New Data/Urban_sound_8k"
    SAMPLE_RATE = 16000

    #This object passed into UrbanSoundDatasetClass
    #ms = melspectrogram(signal) ----> in get item method
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    usd = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset")
    signal, label = usd[0]
