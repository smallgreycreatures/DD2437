import numpy as np

def sample_categorical(probabilities):

    """
    Sample one-hot activations from categorical probabilities

    Args:
      support: shape is (size of mini-batch, number of categories)
    Returns:
      activations: shape is (size of mini-batch, number of categories)
    """

    cumsum = np.cumsum(probabilities,axis=1)
    rand = np.random.random_sample(size=probabilities.shape[0])[:,None]
    activations = np.zeros(probabilities.shape)
    activations[range(probabilities.shape[0]),np.argmax((cumsum >= rand),axis=1)] = 1
    return activations

def sample_binary(on_probabilities):

    """
    Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities

    Args:
      support: shape is (size of mini-batch, size of layer)
    Returns:
      activations: shape is (size of mini-batch, size of layer)
    """

    activations = 1. * ( on_probabilities >= np.random.random_sample(size=on_probabilities.shape) )
    return activations

p = np.array([0.2,0,0,0,0,1,0,1,0,0])
p = p.reshape((1,10))
a = sample_binary(p)
print(a)
