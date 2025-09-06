from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class SimpleTokenizer:
    def __init__(self, language, use_stemming, use_stopwords):
        self.language = language
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self.stop_words = set(stopwords.words(language)) if use_stopwords else set()
        self.stemmer = SnowballStemmer(language) if use_stemming else None
        self.vocabulary = {}
        self.vocab_size = 0

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower(), language=self.language)
        tokens = [token for token in tokens if token.isalpha()]

        if self.use_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
            pos_tags = pos_tag(tokens)
            allowed_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                           'RB', 'RBR', 'RBS'}
            tokens = [word for word, pos in pos_tags if pos in allowed_pos]

        return tokens

    def build_vocabulary(self, texts):
        all_words = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_words.extend(tokens)

        unique_words = sorted(set(all_words))
        self.vocabulary = {word: idx for idx, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)