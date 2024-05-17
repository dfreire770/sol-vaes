import torch
from torch.utils.data import Dataset

from sound_utility import AudioUtil

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path,duration,sr,n_fft, hop_len, label, cutoff=128, highpass=True):
        #sr=44100, n_fft=2048, hop_length=512
        self.df = df
        self.data_path = str(data_path)
        self.duration = duration
        self.cutoff = cutoff
        self.label = label
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.highpass = highpass
            
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'Full_Path']
        #label = self.df.loc[idx,'Instance ID']
        label = self.df.loc[idx,self.label]

        #waveform, spectrogram = AudioUtil.load_and_process_audio(

        spectrogram = AudioUtil.load_and_process_audio(        
            audio_path=audio_file, 
            sr=self.sr,
            n_fft=self.n_fft, 
            duration=self.duration,
            cutoff=self.cutoff,
            highpass=self.highpass)
        
        #return waveform, spectrogram
        #return mel_spectrogram
        #print("spectrogram:",spectrogram.shape)
        #print("label sd:", label)

        return spectrogram, label
    
