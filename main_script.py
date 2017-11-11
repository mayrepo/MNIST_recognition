"""
Script principal qui permet de créer une carte par l'algorithme de kohonen,
de la labelliser et de tester les performances de la reconnaissance (ici de chiffres)
"""

#chargement des données, fonctions et paramètres
import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from load_data import *

from map import map
from label import label
from test import test
from plot_map import plot_map



#===============================================================================
# Création de la carte
#===============================================================================

print("mapping")

#entrainement de la carte
neurons = map(map_shape, data_shape, sigma_max_value, sigma_min_value, eta_max_value, eta_min_value, decay_start_iter, decay_stop_iter, training_samples,initial)

#sauvegarde de la carte
np.save(neurons_path,neurons)




#===============================================================================
# Labellisation de la carte
#===============================================================================

print("labelling")

#labellisation de la carte
neuron_labels = label(map_shape, data_shape, labelling_samples, labelling_labels, neurons)

#sauvegarde das labels
np.save(neuron_labels_path, neuron_labels)




#===============================================================================
# Test des performances
#===============================================================================

print("testing")

#calcul des performances
global_performance, own_performances = test(neurons, neuron_labels, testing_samples, testing_labels)

#affichage des performances dans la console
print("performances : ")
for i, performance in enumerate(own_performances):
    print(i, " : ", round(100*performance,2), "%")

print("total : ", round(100*global_performance,2), "%")


#affichage graphique
plt.subplot(121)
plot_map(neurons,map_shape, data_shape)

plt.subplot(122)
plt.imshow(np.reshape(neuron_labels,map_shape), interpolation='nearest')
plt.colorbar(orientation='horizontal')

plt.show()
