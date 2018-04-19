from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()


"""
configs:
0- No feature
1- sentence polarity
2- num_polarity_words / num_neutral_words
3- mean_pos mean_neg
4- median_pos median_neg
"""


class VaderTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):

        if(self.featureSetConfiguration == 0): # not active
            features = np.array([
                True
                for s in X
            ])
        # sentence_score
        elif self.featureSetConfiguration == 1:
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection = aaecCorpus.sentence
            temp = []
            for sentenceTuple in X:
                features_to_set = {'value': 0}
                # current learning instance info from database
                currentSentence = sentenceCollection.find_one({"$and": [{"articleId": int(sentenceTuple[0])}, {"sentenceId": int(sentenceTuple[1])}]})
                raw_sentence = currentSentence['originalText'].lower()
                b = afinn.score(raw_sentence)
                features_to_set['value'] = b
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])
        # num_polarity_words / num_neutral_words
        elif self.featureSetConfiguration == 2:
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection = aaecCorpus.sentence
            temp = []
            for sentenceTuple in X:
                features_to_set = {'polar/neutral': 0}
        # mean_pos mean_neg
        elif self.featureSetConfiguration == 3:
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection = aaecCorpus.sentence
            temp = []
            for sentenceTuple in X:
                features_to_set = {'polar/neutral': 0}
        # median_pos median_neg
        elif self.featureSetConfiguration == 4:
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection = aaecCorpus.sentence
            temp = []
            for sentenceTuple in X:
                features_to_set = {'polar/neutral': 0}
        return temp

    def fit(self, X, y=None, **fit_params):
        return self

    ## names are related to featureSetConfiguration
    def get_feature_names(self):
        return sorted(AfinnTransformer.features.keys())






