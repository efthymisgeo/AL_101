


(X, y) = download()
(X_train_full, y_train_full, X_test, y_test) = split(trainset_size)
print('train:', X_train_full.shape, y_train_full.shape)
print('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print('unique classes', classes)


def pickle_save(fname, data):
    filehandler = open(fname, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    print('saved', fname, os.getcwd(), os.listdir())


def pickle_load(fname):
    print(os.getcwd(), os.listdir())
    file = open(fname, 'rb')
    data = pickle.load(file)
    file.close()
    print(data)
    return data


def experiment(d, models, selection_functions, Ks, repeats, contfrom):
    algos_temp = []
    print('stopping at:', max_queried)
    count = 0
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
                        alg = TheAlgorithm(k,
                                           model_object,
                                           selection_function
                                           )
                        alg.run(X_train_full, y_train_full, X_test, y_test)
                        d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.accuracies)
                        fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                        pickle_save(fname, d)
                        if count % 5 == 0:
                            print(json.dumps(d, indent=2, sort_keys=True))
                        print()
                        print('---------------------------- FINISHED ---------------------------')
                        print()
    return d


max_queried = 500

repeats = 1

models = [SvmModel, RfModel, LogModel]

selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection]

Ks = [250, 125, 50, 25, 10]

d = {}
stopped_at = -1

# print('directory dump including pickle files:', os.getcwd(), np.sort(os.listdir()))
# d = pickle_load('Active-learning-experiment-' + str(stopped_at) + '.pkl')
# print(json.dumps(d, indent=2, sort_keys=True))

d = experiment(d, models, selection_functions, Ks, repeats, stopped_at + 1)
print(d)
results = json.loads(json.dumps(d, indent=2, sort_keys=True))
print(results)