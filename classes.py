import gensim.downloader
from nltk import word_tokenize
import nltk
import numpy as np
nltk.download('punkt')


# naive bayesian, n-grams, logistic regression, NN
class Benchmark:
    def __init__(self):
        self.pre_trained_embedding_models = ["fasttext-wiki-news-subwords-300", "word2vec-google-news-300", "glove-wiki-gigaword-300"]


# Works for pre_trained_embeddings_models in Benchmark class
class PreTrainedEmbedding:
    def __init__(self, name):
        self.wv = gensim.downloader.load(name)
        print(self.wv.vector_size)

    def __call__(self, sentence):
        words = word_tokenize(sentence)
        return np.mean(np.array([self.wv[word] for word in words]), axis=0)

# class Model():
#     def __init__(self):
#         pass


# embedder = PreTrainedEmbedding(pre_trained_embedding_models[0])

# benchmark avec les modèles au dessus -> prendre le meilleur et faire un grid search, fine tuning
# générer de la donnée 