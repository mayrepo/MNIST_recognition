import numpy as np

import kohonen

def map(map_shape, data_shape, sigma_max_value, sigma_min_value, eta_max_value, eta_min_value, decay_start_iter, decay_stop_iter, training_samples,initiale):
    """
    crée la carte labélisée par l'algorithme de Kohonen
    
    map_shape -- taille de la carte de la forme (lignes, colonnes)
    data_shape -- taille des vecteurs de la forme (lignes, colonnes)
    iterations -- nombre d'échantillon utilisés pour entrainer la carte
    sigma_max_value, sigma_min_value, eta_max_value, eta_min_value, decay_start_iter, decay_stop_iter  -- arguments de la mise à jour de la carte
    training_samples -- tableau des vecteurs utilisés pour entrainer la carte
    """
    
    ## dimensions des vecteurs de poids
    neurons_dimension = (np.prod(map_shape), np.prod(data_shape))

    ## initialisation des prototypes des vecteurs de poids
    neurons = np.zeros(neurons_dimension)
    for i in range(np.prod(map_shape)):
        neurons[i] = initiale
    
    #===============================================================================
    # Boucle d'apprentissage suivant l'algorithme de Kohonen
    #===============================================================================
    for curr_iter, sample in enumerate(training_samples):
        
        ## récupérer les valeurs de sigma et eta
        sigma = kohonen.constrainedExponentialDecay(curr_iter, decay_start_iter, decay_stop_iter*len(training_samples), sigma_max_value, sigma_min_value)
        eta = kohonen.constrainedExponentialDecay(curr_iter, decay_start_iter, decay_stop_iter*len(training_samples), eta_max_value, eta_min_value)
        
        ## trouver la best-matching unit (BMU) et son score (plus petite distance)
        bmu_idx, bmu_score = kohonen.nearestVector(sample, neurons)
        
        ## traduire la position 1D de la BMU en position 2D dans la carte
        bmu_2D_idx = np.unravel_index(bmu_idx, map_shape)
        
        ## gaussienne de taille sigma à la position 2D de la BMU
        gaussian_on_bmu = kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma)
        
        ## mettre à jour les prototypes d'après l'algorithme de Kohonen (fonction à effets de bord)
        kohonen.updateKohonenNeurons(sample, neurons, eta, gaussian_on_bmu)
    
    return neurons

