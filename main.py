import numpy as np
import json
import os
from dataloader.feature_loader import load_feats
from utils.helper_functions import split, experiment

from model.BaseModel import SvmModel, RfModel, LogModel
from model.BaseSelectionFunction import RandomSelection, MarginSamplingSelection, EntropySelection

X, y = load_feats()

trainset_size = 0.85
initial_dataset = split(X, y, trainset_size)
(X_train_full, y_train_full, X_test, y_test) = initial_dataset
print('train:', X_train_full.shape, y_train_full.shape)
print('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print('unique classes', classes)

### active learning rounds
max_queried = 9

repeats = 4
#models = [SvmModel, RfModel, LogModel]
models = [SvmModel]
selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection]

Ks = [50]

d = {}
stopped_at = -1

# print('directory dump including pickle files:', os.getcwd(), np.sort(os.listdir()))
# d = pickle_load('Active-learning-experiment-' + str(stopped_at) + '.pkl')
# print(json.dumps(d, indent=2, sort_keys=True))

d = experiment(d, models, selection_functions, Ks,
               repeats, stopped_at + 1, max_queried,
               initial_dataset)
print(d)
results = json.loads(json.dumps(d, indent=2, sort_keys=True))
print(results)
