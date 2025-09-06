from BoW import BagOfWords
from TF_IDF import TF_IDF

import argparse
from pathlib import Path
import pandas as pd
import nltk

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource == 'stopwords' else f'taggers/{resource}')
        except LookupError:
            nltk.download(resource)

def read_texts_from_file(texts_path, column_names):
    df = pd.read_csv(texts_path)
    X = df[column_names].tolist()
    return X

def read_texts_from_path(texts_path):
    X = []
    y = []
    for path in texts_path.glob("*.txt"):
        with open(path, 'r') as file:
            X.append(file.read())
            y.append(path.name)
    return X, y

def read_texts(texts_path, column_names=None):
    X, y = [], []
    if texts_path.is_file() and texts_path.suffix == ".csv":
        X = read_texts_from_file(texts_path, column_names)

    if texts_path.exists() and not texts_path.is_file():
        X, y = read_texts_from_path(texts_path)

    return X, y

def get_args():
    parser = argparse.ArgumentParser(description='Векторизация текстов с использованием BoW или TF-IDF')

    parser.add_argument('--method', type=str, choices=['bow', 'tfidf'], default='bow',
                        help='Метод векторизации: bow (мешок слов) или tfidf (TF-IDF)')
    parser.add_argument('--input', type=str, required=True,
                        help='Путь к входному CSV файлу с текстами')
    parser.add_argument('--text_column', type=str, required=True,
                        help='Название столбца с текстом в CSV файле')

    parser.add_argument('--language', type=str, default='russian',
                        help='Язык текста (по умолчанию: russian)')
    parser.add_argument('--use_stopwords', action='store_true',
                        help='Использовать стоп-слова')
    parser.add_argument('--use_stemming', action='store_true',
                        help='Использовать стемминг')

    args = parser.parse_args()

    return args

def main():
    download_nltk_resources()
    args = get_args()

    path = Path(args.input)
    columns = args.text_column
    X, y = read_texts(path, columns)
    language = args.language
    use_stopwords = args.use_stopwords
    use_stemming = args.use_stemming

    if args.method == "bow":
        vectorizer = BagOfWords(language, use_stemming, use_stopwords)
    elif args.method == "tfidf":
        vectorizer = TF_IDF(language, use_stemming, use_stopwords, True)
    else:
        return -1

    vector = vectorizer(X)

    print("Улучшенные векторы (без стоп-слов):", vector)
    print("Новый словарь:", vectorizer.vocabulary)

if __name__ == "__main__":
    main()