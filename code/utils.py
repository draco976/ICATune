import random

def disambiguate_prompt(prompt, seq_len, length_X, num_seq):
    """
    Disambiguate the prompt by generating pre-aligned prompt formatting.
    
    Args:
        prompt: Original prompt string
        seq_len: Length of each sequence
        length_X: Length of X part
        num_seq: Number of sequences
        
    Returns:
        Disambiguated prompt
    """
    prompt = prompt.replace('!', '')
    new_prompt = ''
    for i in range(16-num_seq, 16):
        example = prompt[(length_X+3+length_X*seq_len+1)*i: (length_X+3+length_X*seq_len+1)*(i+1)].strip().split(' : ')
        X = example[0]
        Y = [example[1][j:j+seq_len] for j in range(0, len(example[1]), seq_len)]
        new_prompt += X + ' : ' + ''.join([X[k] + ' : ' + Y[k] + ' ' for k in range(len(Y))])

    new_prompt = ''.join(['!'+x if x.isalpha() else x for x in new_prompt])

    return new_prompt    

def sample_cfg(x):
    """
    Sample from context-free grammar to generate a sequence.
    
    Args:
        x: Non-terminal symbol to expand
        
    Returns:
        Generated sequence from grammar
    """
    X = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    if x == 'Z':
        if random.random() < 0.5:
            return sample_cfg('Z1') + sample_cfg('Z2')
        else:
            return sample_cfg('Z2') + sample_cfg('Z1')
    elif x == 'Z1':
        z1 = X[:4]
        random.shuffle(z1)
        return z1
    elif x == 'Z2':
        z2 = X[4:]
        random.shuffle(z2)
        return z2