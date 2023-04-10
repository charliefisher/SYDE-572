import numpy as np
import matplotlib.pyplot as plt
import os

def randMEDClassifier(a_points_subset, b_points_subset) -> None:
    rng = np.random.default_rng()
    
    z_a = rng.choice(a_points_subset)
    z_b = rng.choice(b_points_subset)
        
    # if negative label as a, if positive label as b 
    G_discrim = lambda x: (np.dot(-z_a, x) + 0.5* np.dot(z_a,z_a)) - (np.dot(-z_b, x) + 0.5* np.dot(z_b, z_b))
    
    # for each point in a and b classify using the random prototypes
    a_classified = np.sign([G_discrim(a_point) for a_point in a_points_subset])
    b_classified = np.sign([G_discrim(b_point) for b_point in b_points_subset])
        
    return G_discrim, a_classified, b_classified


def learn_classifiers(a_points, b_points, limit = -1):

    a_points = a_points.astype('float')
    b_points = b_points.astype('float')
    j = 0 
    
    # initialize as an empty sequential classifier
    n_aB = []
    n_bA = []
    G = []

    while a_points.size > 0 and b_points.size > 0 and j != limit: 
        
        # perform random classification and count incorrect in each class 
        G_i, a_label, b_label = randMEDClassifier(a_points, b_points)
        n_aB_i = np.count_nonzero(a_label == 1)
        n_bA_i = np.count_nonzero(b_label ==-1)
                
        if n_aB_i == 0 or n_bA_i ==0:
            # print("Adding sequential classifier %d" % j)
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
    
    return (G, n_aB, n_bA)
            
        
def use_sequential_classify(classifier, X_point):
    j = 0
    
    # from classifier touple, extract sequential classifier elements
    G = classifier[0]
    n_aB = classifier[1]
    n_bA = classifier[2]
        
    while j  < len(G):
        # label individual point 
        label = np.sign(G[j](X_point))
        
        # if label matches a perfectly classified region return label
        if label == 1 and n_aB[j] == 0:
            return label
        if label == -1 and n_bA[j] == 0:
            return label
        
        # else, check the next sequential classifiers  
        j+=1
        
    return label


"""Helper function that creates correct directory for figure, saves it, and closes it"""
def _save_figure(name: str) -> None:
    path = os.path.join('plots', name)
    dir, _ = os.path.split(path)
    os.makedirs(dir, exist_ok=True)
    plt.savefig(path)
    plt.close()
    
    
def test_limited_sequential_classifier(data):
    
    num_limits = 5
    num_tests = 20
    limits = np.arange(num_limits) + 1
    
    # initialize error rates
    error_rates= np.zeros([num_tests, num_limits])
    
    # find total number of points
    num_points = len(data['a']) + len(data['b'])
    
    for limit in limits:
        for i in range(num_tests):
            # learn J-limited sequential classifiers from the train data 
            classifier = learn_classifiers(data['a'], data['b'], limit)
            
            # classify the train data 
            a_label = np.array([use_sequential_classify(classifier, position) for position in data['a']])
            b_label = np.array([use_sequential_classify(classifier, position) for position in data['b']])
            
            # determine the error rate 
            error_rates[i, limit-1] = (np.count_nonzero(a_label == 1) + np.count_nonzero(b_label == -1))/num_points
            
            # Check Error Trends 
            # print("a error rate %f" % (np.count_nonzero(a_label == 1)/200))
            # print("b error rate %f" % (np.count_nonzero(b_label == -1)/200))
            # estimate_sequential_classifier(data, f"j_{limit}_test", classifier=classifier)

    
    plt.figure() 
    mins = error_rates.min(0)
    maxes = error_rates.max(0)
    means = error_rates.mean(0)
    std = error_rates.std(0)
    
    print("Means, Mins, Maxes, STD")
    print(means)
    print(mins)
    print(maxes)
    print(std)

    # create stacked errorbars:
    _, ax = plt.subplots()
    ax.errorbar(limits, means, std, fmt='ok', lw=3, label='Standard Deviation')
    ax.errorbar(limits, means, [means - mins, maxes - means],
             fmt='.k', ecolor='gray', lw=1, label ='Min-Max')
    ax.plot(limits, means, 'b', label = 'Mean')

    ax.legend()
    ax.set_xlim(0.5, num_limits + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$J$')
    ax.set_ylabel('Error (%)')

    _save_figure('part3/{method}.png'.format(method='error_lim'))


            
def estimate_sequential_classifier(data, name, j=-1, classifier = None):
    
    if classifier is None:
        classifier = learn_classifiers(data['a'], data['b'], j)
        
    all_class_data = [data['a'], data['b']]
    N_classes = len(all_class_data)
    
    # construct 2D meshgrid over which to evaluate classifer points 
    grid_resolution = 0.5
    x_lim = (70, 540)
    y_lim = (-10, 430)
    x = np.arange(x_lim[0], x_lim[1] + grid_resolution, grid_resolution)
    y = np.arange(y_lim[0], y_lim[1] + grid_resolution, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # create flattened mshgrid 
    positions = np.vstack([X.ravel(), Y.ravel()]).T    
    
    # classify all meshgrid points and reshape to meshgrid shape
    positions_classifed = np.array([use_sequential_classify(classifier, position) for position in positions]).reshape(X.shape)
    
    # plot data points and boundary
    class_labels = ('Class A', 'Class B')
    class_colors = ('r', 'g')

    _, ax = plt.subplots()
    for i, data in enumerate(all_class_data):
        ax.scatter(data[:, 0], data[:, 1], s=5, c=class_colors[i], label=class_labels[i])
        ax.contour(X, Y, positions_classifed == 1, levels=1, colors='k')
    ax.contourf(X, Y, positions_classifed, 1, colors=(*class_colors, 'w'), alpha=0.1)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
    _save_figure('part3/{method}.png'.format(method=name))

    
        


    