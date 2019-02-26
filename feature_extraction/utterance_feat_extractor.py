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
    utt_id = 0
    for speaker in a:
        for utt in a[speaker]:
            y[utt_id] = emos[a[speaker][utt]['emotion']]
            audio_sig = a[speaker][utt]['wav']
            fs = a[speaker][utt]['Fs']
            dur = a[speaker][utt]['wav_duration']
            mfccs = librosa.feature.melspectrogram(y = audio_sig,
                                                   sr = fs,
                                                   n_fft = 2000,
                                                   hop_length = int(400*dur))
            X[utt_id,:] = mfccs.reshape(-1)
            utt_id += 1
    utt_path = 'extracted_features/utterance'
    filename = os.path.join(utt_path, "X.out.npy")
    save_dir = os.path.abspath(filename)
    #fd = open(save_dir, 'w')
    np.save(save_dir, X)
    #os.close(fd)
    filename = os.path.join(utt_path, "y.out.npy")
    save_dir = os.path.abspath(filename)
    #fd = open(save_dir, 'w')
    np.save(save_dir, y)
    #os.close(fd)
    save_dir = os.path.join(os.pardir, utt_path)
    return(X, y, save_dir)

