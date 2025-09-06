import numpy as np
import re
from collections import Counter

from SimpleTokenizer import SimpleTokenizer


class TF_IDF(SimpleTokenizer):
    def __init__(self, language, use_stemming, use_stopwords, smooth_idf=True):
        super().__init__(language, use_stemming, use_stopwords)
        self.smooth_idf = smooth_idf
    def __tf(self, texts):

        """
        TF(t, d) = (количество раз, когда термин t встречается в документе d) / (общее количество терминов в документе d)
        """

        tf_matrix = np.zeros((len(texts), self.vocab_size), dtype=np.float64)

        for i, text in enumerate(texts):
            tokens = self.preprocess_text(text)
            word_counts = Counter(tokens)

            for word, count in word_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf_matrix[i, idx] = count

            total_words = len(tokens)
            if total_words > 0:
                tf_matrix[i] = tf_matrix[i] / total_words

        return tf_matrix

    def __idf(self, texts):

        """
        IDF(t) = log(общее количество документов / количество документов, содержащих термин t) + 1
        """

        n_docs = len(texts)
        doc_freq = np.zeros(self.vocab_size, dtype=np.int32)

        for text in texts:
            tokens = set(self.preprocess_text(text))
            for word in tokens:
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    doc_freq[idx] += 1

        idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1
        return idf

    def __call__(self, texts):

        """
        TF-IDF(t, d) = TF(t, d) * IDF(t)
        """

        self.build_vocabulary(texts)
        tf_matrix = self.__tf(texts)
        idf_vector = self.__idf(texts)

        tfidf_matrix = tf_matrix * idf_vector

        return tfidf_matrix