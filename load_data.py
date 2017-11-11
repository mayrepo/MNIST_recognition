"""
extrait les données du fichier situé à data_path
"""

data_path = "../Data/mnist.pkl.gz"

#importe les fonctions pour lire les données au format .pkl.gz
import gzip
import pickle

#from parameters import data_path

#décompresse et charge le fichier en mémoire
with gzip.open(data_path, 'rb') as file_handler:
    data = pickle.load(file_handler, encoding='latin1')

## répartit les sous-ensembles (de type 'tuple')
training_set, labelling_set, testing_set = data

## répartit les sous-sous-ensembles (de type 'np.ndarray')
training_samples, training_labels = training_set
labelling_samples, labelling_labels = labelling_set
testing_samples, testing_labels = testing_set

