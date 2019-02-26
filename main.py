import numpy as np
import os
from dataloader.feature_loader import load_feats
from utils.helper_functions import split, experiment

from model.BaseModel import SvmModel, RfModel, LogModel
from model.BaseSelectionFunction import RandomSelection, MarginSamplingSelection, EntropySelection

X, y = load_feats()
############################
### ALCHEMY PART BEGINS ####
# SHOULD BE REMOVED IN ORDER
# TO EXTRACT AGAIN FEATURES
# IN SERVER
###########################
false_idx = np.argmin(y)
print(y[false_idx])
y[false_idx] = 0
print(y[false_idx])
false_idx = np.argmin(y)
print(y[false_idx])
y[false_idx] = 0
print(y[false_idx])
#### ALCHEMY ENDS ########

trainset_size = int(0.7*y.size)
(X_train_full, y_train_full, X_test, y_test) = split(X, y, trainset_size)
print('train:', X_train_full.shape, y_train_full.shape)
print('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print('unique classes', classes)



max_queried = 150

repeats = 1

#models = [SvmModel, RfModel, LogModel]
models = SvmModel
selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection]

Ks = [25, 15, 10, 5]

d = {}
stopped_at = -1

# print('directory dump including pickle files:', os.getcwd(), np.sort(os.listdir()))
# d = pickle_load('Active-learning-experiment-' + str(stopped_at) + '.pkl')
# print(json.dumps(d, indent=2, sort_keys=True))

d = experiment(d, models, selection_functions, Ks, repeats, stopped_at + 1)
print(d)
results = json.loads(json.dumps(d, indent=2, sort_keys=True))
print(results)