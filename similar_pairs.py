import numpy as np
from scipy.spatial.distance import cosine


def cosine_sim(u, v):
    """
    Compute the cosine similarity between vectors of the same dimension
    :param u: vector (numpy array)
    :param v: vector (numpy array)
    :return: cosine similarity
    """
    return 1 - cosine(u, v)


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


def finish_similar_pairs(input_words, embeddings):
    """
    find a word to complete the analogy: A is to B as C is to ?
    :param input_words: list of 2 words on the left hand side and the first word on the right hand side of the analogy
    :param embeddings: pre-trained word vectors
    :return: closest word to complete the analogy
    """
    left_one, left_two, right_one = input_words
    target = embeddings[left_one] - embeddings[left_two]
    max_sim = -1
    closest_word = None
    for w, v in embeddings.items():
        if w in input_words:
            continue
        diff = embeddings[right_one] - v
        curr_sim = cosine_sim(target, diff)
        if curr_sim > max_sim:
            max_sim = curr_sim
            closest_word = w
    return closest_word
