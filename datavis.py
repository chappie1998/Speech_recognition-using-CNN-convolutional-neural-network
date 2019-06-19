import librosa
import librosa.display
import numpy as np

fns = ['data/off/00b01445_nohash_0.wav',
       'data/go/00b01445_nohash_0.wav',
       'data/yes/00f0204f_nohash_0.wav']

spec = []
for filename in fns:
    y, sr = librosa.load(filename)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    spec.append(ps)

librosa.display.specshow(librosa.power_to_db(spec[0], ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
librosa.display.specshow(librosa.power_to_db(spec[1], ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
librosa.display.specshow(librosa.power_to_db(spec[2], ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
