import os
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader.feature_loader import load_feats
from sklearn.preprocessing import StandardScaler

def Split_Dataset(split_size):
    '''Function that reads extracted features
    and returns a random split of size
    (1-split_size) in X_train and split_size
    in X_test
    Returns: 2 tuples (train_data),(test_data)'''
    X, y = load_feats()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=split_size,
                                                    random_state=42)
    return (X_train, y_train), (X_test, y_test)


class StdNormalize(object):

    def normalize(self, X_train, X_val, X_test):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return (X_train, X_val, X_test)

    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)
        return (X_train, X_val, X_test)