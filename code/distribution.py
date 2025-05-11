import numpy as np

def h(i, j, m, n):
    """
    Compute h(i, j, m, n), a similarity function between positions.
    
    Args:
        i: Position in source sequence
        j: Position in target sequence
        m: Length of source sequence
        n: Length of target sequence
        
    Returns:
        Similarity score (negative absolute difference)
    """
    return -abs(i / m - j / n)

def delta(j, i, m, n, p0, lambd):
    """
    Compute probability distribution δ(a_i = j | i, m, n).
    
    Args:
        j: Target position
        i: Source position
        m: Length of source sequence
        n: Length of target sequence
        p0: Probability of a_i = 0 (no alignment)
        lambd: Scaling factor for the similarity function
        
    Returns:
        Probability value
    """
    if j == 0:
        return p0
    elif 0 < j <= n:
        Z_lambda = sum(np.exp(lambd * h(i, k, m, n)) for k in range(1, n + 1))
        return (1 - p0) * np.exp(lambd * h(i, j, m, n)) / Z_lambda
    else:
        return 0

def sample_a_i(i, m, n, p0, lambd):
    """
    Sample a_i (alignment position) given parameters.
    
    Args:
        i: Source position
        m: Length of source sequence
        n: Length of target sequence
        p0: Probability of a_i = 0 (no alignment)
        lambd: Scaling factor for similarity
        
    Returns:
        Sampled alignment position
    """
    probabilities = [delta(j, i, m, n, p0, lambd) for j in range(n + 1)]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize probabilities
    return np.random.choice(range(n + 1), p=probabilities)

def sample_distribution(m, n, p0, lambd):
    """
    Sample from the alignment distribution.
    
    Args:
        m: Number of elements in source sequence
        n: Number of elements in target sequence
        p0: Probability of a_i = 0 (no alignment)
        lambd: Scaling factor for similarity
        
    Returns:
        List of sampled alignment positions
    """
    a = []

    for i in range(1, m + 1):
        a_i = sample_a_i(i, m, n, p0, lambd)
        a.append(a_i)

    return a