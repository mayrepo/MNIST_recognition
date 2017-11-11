#===============================================================================
# Algorithme de test des résultats suivant testing: on cherche le pourcentage de bonnes réponses données par la machine
#===============================================================================

import numpy as np

from kohonen import nearestVector

def test(weights, matrice_labelisee, testing_samples, testing_labels):
    """
    évalue la qualité de la décision par l'algorithme de kohonen
    weights -- tableau des neurones
    matrice_labelisee -- tableau contenant les labels associés aux neurones
    testing_samples -- échantillons utilisés pour faire l'évaluation
    testing_labels -- labels réels associés aux échantillons
    """
    
    #variables comptant le nombre de bonnes réponse (globalement et selon les nombres)
    success = 0
    successes = np.zeros(10)
    
    number_tests = np.zeros(10)
    
    for sample, label in zip(testing_samples,testing_labels):
        
        #récupère l'indice du neurone le plus proche
        nearest = nearestVector(sample, weights)[0]
        
        number_tests[label] += 1
        
        #si l'estimation est bonne, incrémente les variables correspondantes
        if matrice_labelisee[nearest] == label:
            success +=1
            successes[label] += 1
    
    #retourne les performances globale et selon les nombres
    return (success / sum(number_tests), successes/number_tests)

