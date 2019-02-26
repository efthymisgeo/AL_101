import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import os

data_load_path = os.path.abspath(os.path.join(os.pardir,
                                              'extracted_features/utterance'))
X = np.load(data_load_path + '/X.out.npy')
y = np.load(data_load_path + '/y.out.npy')
y_sc = y.astype(int)
X_scaled = StandardScaler().fit_transform(X)

lin_clf = svm.LinearSVC()
clf = svm.SVC(gamma='scale')
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X_scaled, y_sc, cv=cv, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#lin_clf.fit(X, Y)
#cross_val_score(clf, X, y, cv=cv)
kke = 3
#n_samples = iris.data.shape[0]