import os
import numpy as np
from feature_extraction.utterance_feat_extractor import FeatureExtractor

def load_feats():
    FEATURE_PATH = os.path.abspath(os.path.join(os.pardir,
                                            'AL_101/extracted_features/utterance'))
    if not os.listdir(FEATURE_PATH):
        print("Features not yet extracted")
        X, y, save_dir = FeatureExtractor()
        print("Features Stored in: ", save_dir)
    else:
        print("Features already extracted")
        X = np.load(FEATURE_PATH + '/X.out.npy')
        y = np.load(FEATURE_PATH + '/y.out.npy').astype(int)

    return X, y

if __name__ == "__main__":
    """!brief Example of usage"""
    X, y = load_feats()
    print("Features succesfully loaded")