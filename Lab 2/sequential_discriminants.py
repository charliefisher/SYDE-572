import numpy as np
# from numpy.random import Generator as rng

def learn_classifiers(a_points, b_points, limit = 0) -> None:

    a_points = a_points.astype('float')
    b_points = b_points.astype('float')
    j = 1 
    
    n_aB = []
    n_bA = []
    G = []

    while a_points.size > 0 and b_points.size > 0 and j != limit: 
        
        # perform random classification and count nyincorrect 
        G_i, a_label, b_label = randMEDClassifier(a_points, b_points)
        n_aB_i = np.count_nonzero(a_label == 1)
        n_bA_i = np.count_nonzero(b_label ==-1)
                
        if n_aB_i == 0 or n_bA_i ==0:
            G.append(G_i)
            n_aB.append(n_aB_i)
            n_bA.append(n_bA_i)
            
            j+= 1
            
            if n_aB_i == 0:
                # remove points from b that G classifies as B (only keep A classification)
                b_points = np.delete(b_points, np.where(b_label ==  1), axis = 0)
            if n_bA_i == 0:
                # remove points from a that G classifies as A (only keep B classification)
                a_points = np.delete(a_points, np.where(a_label ==  -1), axis = 0)
    
    print(G)
    print(n_aB)
    print(n_bA)
    return (G, n_aB, n_bA)
            
        
def sequential_classify(classifier, X):
    j = 1 
    
    while j - 1 < classifier[0].shape:
        label = np.sign(classifier[0][j](X_point))
        if label == -1 and clasifier[1][j] == 0:
            return -1
        elif label == 1 and classifier[2][j] == 0:
            return 1 
        
        j+=1
    
    return label


def plot_results(classifier):
    plt.figure()
    
        


def randMEDClassifier(a_points_subset, b_points_subset) -> None:
    rng = np.random.default_rng()
    
    z_a = rng.choice(a_points_subset)
    z_b = rng.choice(b_points_subset)
        
    # if negative label as a, if positive label as b 
    G_discrim = lambda x: (np.dot(-z_a, x) + 0.5* np.dot(z_a,z_a)) - (np.dot(-z_b, x) + 0.5* np.dot(z_b, z_b))
    
    a_classified = np.sign([G_discrim(a_point) for a_point in a_points_subset])
    b_classified = np.sign([G_discrim(b_point) for b_point in b_points_subset])
        
    return G_discrim, a_classified, b_classified
    