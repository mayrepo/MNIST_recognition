#affichage d'une carte

import matplotlib.pyplot as plt
import numpy as np

from plot_image import plot_image

def plot_map(map, map_shape, data_shape):
    """
    affiche la carte
    map -- liste des neurones à afficher
    map_shape -- taille la carte des vecteurs de la forme (ligne, colonne)
    data_shape -- taille des neurones de la forme (ligne, colonne)
    """
    
    #crée la matrice composée des images
    
    #crétion de la matrice vide
    map_2D = np.zeros([0,map_shape[1]*data_shape[1]])
    
    for i in range(map_shape[0]):
        
        #crétion d'une ligne vide d'épaisseur la hauteur d'un neurone
        line = np.zeros([data_shape[0],0])
        
        for j in range(map_shape[1]):
            
            #extraction du neurone de coordonnées (i, j)
            neuron = np.reshape(map[map_shape[1]*i+j],data_shape)
            
            #ajout du neurone à la fin de la ligne
            line = np.concatenate((line, neuron), axis = 1)
            
        #ajout de la ligne à la matrice
        map_2D = np.concatenate((map_2D, line), axis = 0)
        
    
    #map_2D=[[np.rint(256*map[map_shape[1]*(i//data_shape[0])+j//data_shape[1]])[(i%data_shape[0])*data_shape[1] + j%data_shape[1]] for j in range(map_shape[1]*data_shape[1])] for i in range(map_shape[0]*data_shape[0])]
    
    #affiche la matrice
    plt.imshow(np.rint(256*map_2D), cmap='Greys', interpolation='nearest')
