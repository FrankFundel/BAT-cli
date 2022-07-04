import librosa
import numpy as np
from scipy import signal

sample_rate = 22050
n_fft = 512

b, a = signal.butter(10, 15000 / 120000, 'highpass')

def denoise(x):
    return np.abs(x - x.mean())

def prepareData(y):
    filtered = signal.lfilter(b, a, y)                      # filter
    D = librosa.stft(filtered, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)   # spectrogram
    S_db = np.apply_along_axis(denoise, axis=1, arr=S_db)   # denoise
    return np.transpose(S_db)