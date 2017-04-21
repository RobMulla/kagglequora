#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz

from collections import Counter
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import word_tokenize, ngrams


class PipelineEstimator(BaseEstimator, TransformerMixin):
    """Define the necessary methods"""

    def __init__(self):
        pass


    def fit(self, X, y = None):
        return self


    def transform(self, X, y = None):
        return X


class FuzzyFeatures(PipelineEstimator):
    """Parse datetime into its component parts"""

    def __init__(self):
        pass


    def transform(self, X, y = None):
        X['fuzz_qratio'] = X.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        X['fuzz_WRatio'] = X.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        X['fuzz_partial_ratio'] = X.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        X['fuzz_partial_token_set_ratio'] = X.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        X['fuzz_partial_token_sort_ratio'] = X.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        X['fuzz_token_set_ratio'] = X.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        X['fuzz_token_sort_ratio'] = X.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        return X

class WordMatchShare(PipelineEstimator):
    """Return only specified columnss"""

    def __init__(self, cols = (), invert = False):
        self.stops = set(stopwords.words("english"))

    @staticmethod
    def get_weight(count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def word_match_share(self,row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R

    def tfidf_word_match_share(self,row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [self.weights.get(w, 0) for w in q1words.keys() if w in q2words] + [self.weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [self.weights.get(w, 0) for w in q1words] + [self.weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def transform(self, X, y = None):
        qs = pd.Series(X['question1'].tolist() + X['question2'].tolist()).astype(str)
        words = (" ".join(qs)).lower().split()
        counts = Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}

        X['tfidf_word_match_share'] = X.apply(self.tfidf_word_match_share, axis=1, raw=True)

        X['word_match_share'] = X.apply(self.word_match_share, axis=1, raw=True)

        return X


class SentimentFeatures(PipelineEstimator):
    """Parse datetime into its component parts"""

    def __init__(self):
        pass


    def transform(self, X, y = None):
        sid = SentimentIntensityAnalyzer()
        X['q1_polarity'] = X.question1.apply(lambda row: sid.polarity_scores(str(row))['compound'])
        X['q2_polarity'] = X.question2.apply(lambda row: sid.polarity_scores(str(row))['compound'])
        return X

class SelectCols(PipelineEstimator):
    """Return only specified columnss"""

    def __init__(self, cols = (), invert = False):
        self.cols = cols
        self.invert = invert


    def transform(self, X, y = None):
        mask = np.array([True if col in self.cols else False for col in X.columns])
        if self.invert:
            mask = np.invert(mask)
        return X.loc[:, mask]

class LengthShare(PipelineEstimator):
    """Length Share"""

    def __init__(self):
        pass


    def normalized_word_share(self, row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

    def transform(self, X, y = None):
        X['q1len'] = X['question1'].str.len()
        X['q2len'] = X['question2'].str.len()
        # lets calculate difference in question length as well
        X['diff_len'] = X['q1len'] - X['q2len']


        xfill = X.fillna("")
        X['q1_n_words'] = xfill['question1'].apply(lambda row: len(row.split(" ")))
        X['q2_n_words'] = xfill['question2'].apply(lambda row: len(row.split(" ")))
        X['diff_n_words'] = X['q1_n_words'] - X['q2_n_words']

        X['word_share'] = X.apply(self.normalized_word_share, axis=1)

        return X


class BinarySplitter(PipelineEstimator):
    """Binarize a feature and add that as a new feature"""

    def __init__(self, col, threshold, new_name = None):
        """Split col based on threshold"""
        self.col = col
        self.threshold = threshold
        self.new_name = new_name or col + "_bin"


    def transform(self, X, y = None):
        X[self.new_name] = X[self.col] >= self.threshold
        return X

class BinarySplitter(PipelineEstimator):
    """Binarize a feature and add that as a new feature"""

    def __init__(self, col, threshold, new_name = None):
        """Split col based on threshold"""
        self.col = col
        self.threshold = threshold
        self.new_name = new_name or col + "_bin"


    def transform(self, X, y = None):
        X[self.new_name] = X[self.col] >= self.threshold
        return X
