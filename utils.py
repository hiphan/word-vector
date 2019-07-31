import numpy as np


def load_word_vector(num_tokens: int, dimension=300):
    """
    load pre-trained GloVe embeddings
    :param num_tokens: number of tokens used to train word vectors
    :param dimension: dimension of word vectors
    :return: dictionary mapping word to its vector
    """
    path = 'GloVe/glove' + str(num_tokens) + 'B' + str(dimension) + '.txt'
    f = open(path, 'r')
    glove = {}
    for line in f:
        split_line = line.split()
        word = ''.join(split_line[0:len(split_line) - 300])
        if word in glove.keys():
            continue
        embedding = np.asarray(split_line[len(split_line) - 300:], dtype='float32')
        glove[word] = embedding
    f.close()
    return glove

