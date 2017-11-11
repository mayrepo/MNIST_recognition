"""
définition de tous paramètres
"""

import numpy as np
#from compute_average import compute_average

#chemins des fichiers de sauvegarde
average_path = "../Data/average.npy"
neurons_path = "../Data/neurons.npy"
neuron_labels_path = "../Data/neuron_labels.npy"

#taille des neurones
data_shape = (28, 28)

#taille de la carte
map_shape = (10, 10)

#paramètres de l'algorithme de kohonen
sigma_max_value = 3
sigma_min_value = .9

eta_max_value = 3
eta_min_value = .1

decay_start_iter = 0.2
decay_stop_iter = 0.6

#contenu initial des neurones
initial = np.zeros(np.prod(data_shape))

#on ouvre le fichier qui contient la moyenne s'il existe, sinon on le crée
try:
    average = np.load(average_path)
except IOError:
    from load_data import training_samples
    average = np.mean(training_samples)
    np.save(average_path, average)

#contenu initial des neurones : vecteur nul ou vecteur moyen
initial = np.zeros(np.prod(data_shape))
#initial = average

