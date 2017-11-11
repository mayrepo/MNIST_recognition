#affichage d'une image

import matplotlib.pyplot as plt
import numpy as np
from parameters import data_shape

def plot_image(image):
    """
    affiche l'image
    image -- vecteur image
    """
    plt.imshow(np.rint(256*image).reshape(data_shape), cmap='Greys', interpolation='nearest')
