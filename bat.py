#!/usr/bin/env python3

import argparse
import os
from tqdm import tqdm
import functools
import torch
import librosa
import numpy as np
import csv

from datasets.prepare_data import prepareData
from datasets.prepare_sequences import getSequences, germanBats
from models.bat_2 import BAT

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Automatic bat call classification')
parser.add_argument('path', type=dir_path, help='Path to directory where audio files are located (required)')
parser.add_argument('--threshold', type=float, help='Threshold for prediction (if -1 threshold will be determined automatically).', default=0.5)
args = parser.parse_args()

classes = germanBats
sample_rate = 22050          # recordings are in 96 kHz, 24 bit depth, 1:10 TE (mic sr 960 kHz), 22050 Hz = 44100 Hz TE

model = BAT(
    max_len=60,
    patch_dim=44 * 257,
    d_model=64,
    num_classes=len(list(classes)),
    nhead=2,
    dim_feedforward=32,
    num_layers=2,
    seq=False
)
model.load_state_dict(torch.load('models/bat_2_convnet_mixed.pth', map_location='cpu'))
model.eval()
    
def predict(filename):
    y, _ = librosa.load(filename, sr=sample_rate*10)  # expand
    D = librosa.stft(y, n_fft=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    S_db = prepareData(y) # filter, spectrogram, denoise
    
    sequence = np.asarray(getSequences(S_db, patch_len=44, patch_skip=22, seq_len=60, seq_skip=60))
    b, n, w, h = sequence.shape
    input_seq = torch.Tensor(sequence).reshape((b, n * w, h))
    output = model(input_seq)
    prediction = torch.sigmoid(output).mean(axis=0)
    
    def compare(a, b):
        if prediction[a] < prediction[b]:
            return 1
        elif prediction[a] > prediction[b]:
            return -1
        else:
            return 0
    
    if args.threshold == -1:
      threshold = prediction.mean(axis=0)
    else:
      threshold = args.threshold

    labels = torch.nonzero(prediction > threshold)[:, 0].tolist()
    labels.sort(key=functools.cmp_to_key(compare))
    return prediction.tolist(), labels

data = []
for filename in tqdm(os.listdir(args.path)):
    filepath = os.path.join(args.path, filename)
    prediction, labels = predict(filepath)
    row = [filepath]
    for l in labels:
        row.append(list(classes)[l])
        row.append(prediction[l])
    data.append(row)

with open('BAT.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerows(data)

