import pickle


def save_models(classifiers_and_nb):
    filename_list = []
    for idx, classifier in enumerate(classifiers_and_nb):
        filename = f'models_saved/trained_model_{idx}.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        filename_list.append(filename)
    return filename_list


def load_models(filenames):
    all_loaded_models = []
    for filename in filenames:
        loaded_model = pickle.load(open(filename, 'rb'))
        all_loaded_models.append(loaded_model)
    return all_loaded_models
