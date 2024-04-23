import gensim.downloader
from nltk import word_tokenize
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from load_data import get_data
# import nltk
# nltk.download('punkt')

file_path = './data/all.jsonl'


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

model = make_pipeline(CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 1)), TfidfTransformer(), LogisticRegression(random_state=42, class_weight='balanced', penalty='l2', max_iter=500))

X_train, X_test, y_train, y_test = get_data(file_path)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))