import numpy as np

from kohonen import nearestVector

def label(map_shape, data_shape, labelling_samples, labelling_labels, neurons):
    """labellise la carte
    
    map_shape -- taille de la carte de la forme (lignes, colonnes)
    data_shape -- taille des vecteurs de la forme (lignes, colonnes)
    labelling_samples -- échantillons pour labelliser
    labelling_labels -- labels associés aux échantillons
    neurons -- carte des neurones
    neuron_labels -- carte des labels associés aux neurones
    """
    
    #Crée un compteur comme tableau de zéros de 100 lignes et 10 colonnes 
    dimension_compteur=(np.prod(map_shape), 10)
    compteur=np.zeros(dimension_compteur)
    
    
    #met à jour la variable compteur
    for sample, label in zip(labelling_samples, labelling_labels):
        
        #cherche la BMU et sa distance associée
        (nearestVector_coor, nearestVector_dist) = nearestVector(sample, neurons)
        
        #Met à jour le compteur: ajoute 1 au compteur pour les coordonnées BMU associés seulement
        compteur[nearestVector_coor, label]+=1
    
    
    #génère la carte des labels
    neuron_labels=np.zeros(np.prod(map_shape))
    
    for case in range(np.prod(map_shape)):
        #attribue à la case la plus fréquente valeur du compteur 
        neuron_labels[case]=np.argmax(compteur[case,:])

    return neuron_labels

