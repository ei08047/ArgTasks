from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from OPAM.Verb import Verb
from sklearn.preprocessing import MinMaxScaler

'''
featureSetConfiguration
0 : empty
1 : features = {'communication':Bool, 'mental':Bool}
2 : features = {'communication:int, mental: int}
'''

class VerbTypeTransformer(TransformerMixin, BaseEstimator):
    features = {'communication':False, 'mental':False}

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):

        v = Verb()
        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        aaecCorpus = mongoClient.AAECCorpus
        # Sentence's table
        sentenceCollection = aaecCorpus.sentence

        if (self.featureSetConfiguration == 0):  # not active
            features = np.array([
                VerbTypeTransformer.features
                for s in X
            ])
        elif self.featureSetConfiguration == 1:  # active
            temp = []
            for sentenceTuple in X:
                currentSentence = sentenceCollection.find_one({"$and": [{"articleId": int(sentenceTuple[0])}, {"sentenceId": int(sentenceTuple[1])}]})
                raw_sentence = currentSentence['originalText'].lower()
                features = {'communication':False, 'mental':False}
                features['communication'] = v.is_sentence_with_communication_verb(raw_sentence)
                features['mental'] = v.is_sentence_with_mental_verb(raw_sentence)
                features_to_add = [features[key] for key in sorted(features.keys())]
                temp.append(features_to_add)
            return np.array(temp)
        elif self.featureSetConfiguration == 2:
            temp = []
            for sentenceTuple in X:
                currentSentence = sentenceCollection.find_one({"$and": [{"articleId": int(sentenceTuple[0])}, {"sentenceId": int(sentenceTuple[1])}]})
                raw_sentence = currentSentence['originalText'].lower()
                features = {'communication':False, 'mental':False}
                features['communication'] = v.count_communication_verb_in_sentence(raw_sentence)
                features['mental'] = v.count_mental_verb_in_sentence(raw_sentence)
                features_to_add = [features[key] for key in sorted(features.keys())]
                temp.append(features_to_add)
            min_max_scaler = MinMaxScaler()
            temp = min_max_scaler.fit_transform(temp)
        mongoClient.close()
        return np.array(temp)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['communication','mental']
