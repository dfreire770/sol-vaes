import torch
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import numpy as np

class AudioUtil:
    
    def load_and_process_audio(audio_path, duration=None, sr=44100, n_fft=2048, hop_length=512, cutoff=128, highpass=True):
        """
        Load an audio file, trim it if required to a certain duration,
        and return its spectrogram.

        Parameters:
            audio_path (str): Path to the audio file.
            duration (float): Desired duration of the audio (in seconds). If None, no trimming is done.
            sr (int): Sample rate.
            n_fft (int): Number of samples per FFT.
            hop_length (int): Hop length for STFT.

        Returns:
            np.ndarray: Spectrogram of the audio.
        """
        # Load audio file
        waveform, orig_sr = librosa.load(audio_path, sr=sr)

        
        # If duration is specified, trim or pad the audio
        if duration is not None:
            # Compute the number of samples required for the desired duration
            target_length = int(duration * sr)
            current_length = len(waveform)

            # If the audio is shorter than the desired duration, pad with silence
            if current_length < target_length:
                waveform = np.pad(waveform, (0, target_length - current_length), mode='constant')

            # If the audio is longer than the desired duration, trim
            elif current_length > target_length:
                waveform = waveform[:target_length]

        # apply a high pass filter to eliminate low frequencies. (Potential HUM presence)

        if highpass:
            # Define the cutoff frequency for the high pass filter (adjust as needed)
            cutoff_freq = cutoff
        
            nyq = 0.5 * sr
            high = cutoff_freq / nyq
            b, a = butter(2, high, btype='highpass')
            # Apply the filter
            waveform = filtfilt(b, a, waveform)

        # Compute the spectrogram
        #Short-time Fourier transform (STFT)
        #spectrogram = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
        spectrogram = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))


        #print('spectrogram:',spectrogram.shape)

        #return mel_spectrogram
        #frequency_bins, time_steps
        return spectrogram
        #, spectrogram_tensor

    # n_ftt=2048, hop_length=512, 

    def griffin_lim_librosa(self,magnitude_spectrogram, iterations=30, n_fft=2048, hop_length=512, win_length=None, length=None):
        """
        Implements the Griffin-Lim algorithm to estimate the phase given only the magnitude
        of the Short-Time Fourier Transform (STFT).

        Args:
            magnitude_spectrogram (np.ndarray): Magnitude spectrogram
            iterations (int): Number of iterations
            n_fft (int): Number of samples per FFT
            hop_length (int): Hop length for STFT
            win_length (int or None): Window length. If None, defaults to n_fft
            length (int or None): Length of the output signal. If None, it follows librosa's default behavior.
            
        Returns:
            np.ndarray: Reconstructed time-domain signal
        """
        # Randomly initialize phase
        # Randomly initialize phase
        random_phase = 2 * np.pi * np.random.rand(*magnitude_spectrogram.shape)
        phase_real = np.cos(random_phase)
        phase_imag = np.sin(random_phase)

        # Combine real and imaginary parts into a complex tensor
        phase = phase_real + 1j * phase_imag

        # Window function for the STFT/ISTFT
        window = 'hann' if win_length is None else np.hanning(win_length)

        for _ in range(iterations):
            # Inverse STFT
            signal = librosa.istft(magnitude_spectrogram * phase, hop_length=hop_length, win_length=win_length, window=window, length=length)

            # Re-calculate STFT
            complex_spectrogram = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)

            # Update only the phase
            _, phase = librosa.magphase(complex_spectrogram)

        # Final iSTFT to get the time domain signal
        signal = librosa.istft(magnitude_spectrogram * phase, hop_length=hop_length, win_length=win_length, window=window, length=length)
        
        return signal

    def plot_spectrogram(spectrogram_tensor, sr=44100, hop_length=512):
        """
        Plot the spectrogram from a tensor.

        Parameters:
            spectrogram_tensor (np.ndarray): Spectrogram tensor of shape (freq_bins, time_frames).
            sr (int): Sample rate.
            hop_length (int): Hop length for STFT.
        """
        #spectrogram_tensor = np.transpose(spectrogram_tensor)

        # Convert spectrogram to dB scale
        spectrogram_db = librosa.amplitude_to_db(spectrogram_tensor, ref=np.max)

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

