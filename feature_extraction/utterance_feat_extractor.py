import numpy as np
import librosa
import os

print(os.path.dirname(os.path.realpath(__file__)))

from dataloader.dataloader_emodb import EmodbDataLoader

def FeatureExtractor():
    dtldr = EmodbDataLoader()
    a = dtldr.data_dict
    emos = {'anger': 0,
            'boredom': 1,
            'disgust': 2,
            'anxiety/fear': 3,
            'happiness': 4,
            'sadness': 5,
            'neutral': 6}
    y = np.zeros((535,), dtype=int)
    X = np.zeros((535, 128*41))
    for speaker in a:
        for i,utt in enumerate(a[speaker]):
            y[i] = emos[a[speaker][utt]['emotion']]
            audio_sig = a[speaker][utt]['wav']
            fs = a[speaker][utt]['Fs']
            dur = a[speaker][utt]['wav_duration']
            mfccs = librosa.feature.melspectrogram(y = audio_sig,
                                                   sr = fs,
                                                   n_fft = 2000,
                                                   hop_length = int(400*dur))
            X[i,:] = mfccs.reshape(-1)
    save_dir = os.path.abspath(os.path.join(os.pardir, 'extracted_features/utterance'))
    np.save(save_dir + '/X.out', X)
    np.save(save_dir + '/y.out', y)
    return(X, y, save_dir)

