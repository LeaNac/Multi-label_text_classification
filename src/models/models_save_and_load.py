import pickle


def save_models(models, vectorizer):
    #TODO create folder automatically
    for model_name in models.keys():
        filename = f'models/model_{model_name}.pkl'
        pickle.dump(models[model_name], open(filename, 'wb'))
    filename = f'models/vectorizer.pkl'
    pickle.dump(vectorizer, open(filename, 'wb'))


def load_models(models_filenames, vectorizer_filename):
    all_loaded_models = {}
    for idx, filename in enumerate(models_filenames):
        model_class = models_filenames[idx].split('/')[-1].split('.')[0].split('model_')[1]
        all_loaded_models[model_class] = pickle.load(open(filename, 'rb'))
    vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
    return all_loaded_models, vectorizer