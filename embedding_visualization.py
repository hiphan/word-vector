import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize(list_of_words, embeddings, embedding_dimension=300, n_components=2):
    """
    Reduce the dimension of word embeddings to visualize using t-SNE algorithm
    :param list_of_words: list of word to visualize
    :param embeddings: pre-trained GloVe embeddings
    :param embedding_dimension: dimension of word embeddings. Default 300
    :param n_components: dimension to reduce to. Default 2
    :return: transformed embeddings
    """
    num_word = len(list_of_words)

    # put embeddings in a numpy matrix
    e_matrix = np.zeros((num_word, embedding_dimension))
    for idx, w in enumerate(list_of_words):
        vec = embeddings[w]
        e_matrix[idx] = vec

    # reduce dimension
    transformed = TSNE(n_components=n_components).fit_transform(e_matrix)

    # visualize
    fig, ax = plt.subplots()
    ax.scatter(transformed[:, 0], transformed[:, 1])
    for idx, w in enumerate(list_of_words):
        ax.annotate(w, (transformed[idx, 0], transformed[idx, 1]))
    plt.plot()
