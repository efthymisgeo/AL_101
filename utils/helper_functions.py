import numpy as np
import pickle
import os
from model.ActiveRound import TheAlgorithm
from sklearn.model_selection import train_test_split

def split(X, y, train_size):
    '''' Function that shuffles and then
    splits dataset Returns Train and
    Test Dataset'''
    X_train_full, X_test, y_train_full, y_test = \
        train_test_split(X, y, test_size = 1-train_size,
                         random_state=69)
    return (X_train_full, y_train_full, X_test, y_test)

def pickle_save(fname, data):
    '''function that saves current model's
    accuracies et. all'''
    filehandler = open(fname, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    print('saved', fname, os.getcwd(), os.listdir())

def pickle_load(fname):
    '''function that loads previous accuracies
    in order to resume training'''
    print(os.getcwd(), os.listdir())
    file = open(fname, 'rb')
    data = pickle.load(file)
    file.close()
    print(data)
    return data



def experiment(d, models, selection_functions,
               Ks, repeats, contfrom, max_queried,
               initial_dataset):
    '''This is the main script that is run in every AL round
    INPUTS
     d : model dictionary
     models : list of models to be run
     selection_functions : list of AL strategies (the heart of our experiment)
     Ks : list of k-sample pooling in every AL round
     repeats :  int
     contfrom:
     max_queried: max_samples queried '''
    (X_train_full, y_train_full, X_test, y_test) = initial_dataset
    algos_temp = []
    print('stopping at:', max_queried)
    count = 0

    #loops through every possible model
    #using every possible AL strategy for
    #all given number of different k-sampling

    for model_object in models:
        if model_object.__name__ not in d:
            d[model_object.__name__] = {}

        for selection_function in selection_functions:
            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}

            for k in Ks:
                d[model_object.__name__][selection_function.__name__][str(k)] = []

                for i in range(0, repeats):
                    count += 1
                    if count >= contfrom:
                        print('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (
                        count, model_object.__name__, selection_function.__name__, k, i))
                        alg = TheAlgorithm(k,model_object,selection_function)
                        alg.run(X_train_full, y_train_full, X_test, y_test, max_queried)
                        d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.accuracies)
                        fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                        pickle_save(fname, d)
                        if count % 5 == 0:
                            print(json.dumps(d, indent=2, sort_keys=True))
                        print()
                        print('---------------------------- FINISHED ---------------------------')
                        print()
    return d