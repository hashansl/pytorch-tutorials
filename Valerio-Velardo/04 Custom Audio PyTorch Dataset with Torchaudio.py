#04 Uses urban sound 8K dataset

from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os


class UrbanSoundDataset(Dataset):

    def __init__(self,annotation_file, audio_dir):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir

    def __len__(self):
        # len(usd)
        return len(self.annotations)

    def __getitem__(self, index):
        # a_list[1] --> a_list.__getitem__(1) <----Under the hood
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        return signal,label

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])

        return path

    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index, 6]

if __name__ == "__main__":

    ANNOTATION_FILE = "/media/hashan/Laptop Hard Bk/python projects-B/Research/Sound/jupyter/Audio_classification_iron/New Data/Urban_sound_8k/UrbanSound8K.csv"
    AUDIO_DIR = "/media/hashan/Laptop Hard Bk/python projects-B/Research/Sound/jupyter/Audio_classification_iron/New Data/Urban_sound_8k"

    usd = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dataset")

    signal, label = usd[0]





# annotation = pd.read_csv("/media/hashan/Laptop Hard Bk/python projects-B/Research/Sound/jupyter/Audio_classification_iron/New Data/Urban_sound_8k/UrbanSound8K.csv")
# print(annotation.head())
# print(len(annotation))

