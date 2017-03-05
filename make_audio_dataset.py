import os
import subprocess

AUDIO_DIR = 'audio/jokes'
NONAUDIO_DIR = 'audio/nonjokes'

for fname in os.listdir(AUDIO_DIR):
    if fname.endswith('.mp3') or fname.endswith('.wav'):
        fpath = os.path.join(AUDIO_DIR, fname)
        output_fpath = os.path.join(AUDIO_DIR, 'processed', fname[:-4] + '.txt')
        # aubiopitch --input /Users/valentin/Downloads/Bar_Dog.mp3 -H 2000 -r 10000
        f = open(output_fpath, 'w')
        subprocess.call(['aubiopitch', '--input', fpath, '-H', '1024', '-r', '10000'], stdout=f)

for fname in os.listdir(NONAUDIO_DIR):
    if fname.endswith('.mp3') or fname.endswith('.wav'):
        fpath = os.path.join(NONAUDIO_DIR, fname)
        output_fpath = os.path.join(NONAUDIO_DIR, 'processed', fname[:-4] + '.txt')
        # aubiopitch --input /Users/valentin/Downloads/Bar_Dog.mp3 -H 2000 -r 10000
        f = open(output_fpath, 'w')
        subprocess.call(['aubiopitch', '--input', fpath, '-H', '1024', '-r', '10000'], stdout=f)
