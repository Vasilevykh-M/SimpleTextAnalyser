import numpy as np
import re
from collections import Counter

from SimpleTokenizer import SimpleTokenizer


class BagOfWords(SimpleTokenizer):
    def __init__(self, language, use_stemming, use_stopwords):
        super().__init__(language, use_stemming, use_stopwords)

    def __transform(self, texts):
        vectors = np.zeros((len(texts), self.vocab_size), dtype=np.int32)

        for i, text in enumerate(texts):
            tokens = self.preprocess_text(text)
            word_counts = Counter(tokens)

            for word, count in word_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    vectors[i, idx] = count

        return vectors

    def __call__(self, texts):
        self.build_vocabulary(texts)
        return self.__transform(texts)