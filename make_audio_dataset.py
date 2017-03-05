import os
import subprocess
from vad import VAD
import numpy as np
from scipy.io.wavfile import read

AUDIO_DIR = 'audio/jokes'
NONAUDIO_DIR = 'audio/nonjokes'


def decode(fname):
    a = read(fname)
    return np.array(a[1], dtype=float)

for fname in os.listdir(AUDIO_DIR):
    if fname.endswith('.mp3') or fname.endswith('.wav'):
        fpath = os.path.join(AUDIO_DIR, fname)
        a = decode(fpath)
        detector = VAD(fs=16000)
        speech = detector.detect_speech(a, threshold=0.1)
        print('You ugly, you your daddys son.')

for fname in os.listdir(NONAUDIO_DIR):
    if fname.endswith('.mp3') or fname.endswith('.wav'):
        fpath = os.path.join(NONAUDIO_DIR, fname)
