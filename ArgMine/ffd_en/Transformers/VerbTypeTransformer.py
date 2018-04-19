from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from sklearn.preprocessing import MinMaxScaler
from OPAM.Verb import Verb

'''
featureSetConfiguration
0 : empty
1 : features = {'communication':Bool, 'mental':Bool}
2 : features = {'communicationCount':Int ,'mentalCount': Int}
'''

class VerbTypeTransformer(TransformerMixin, BaseEstimator):
    features = {'communication':False, 'mental':False}


    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        mongoClient = MongoClient('localhost', 27017)
        ffCorpus = mongoClient.FACTFEELCorpus
        documentCollection = ffCorpus.documents

        temp = []
        if (self.featureSetConfiguration == 0):  # not active
            temp = [
                [0 for f in VerbTypeTransformer.features.keys()]
                for s in X
            ]

        elif self.featureSetConfiguration == 1:  # active
            v = Verb()
            # connect to db
            for document in X:
                currentSentence = documentCollection.find_one({'document_id': document })
                raw_sentence = currentSentence['raw'].lower()
                features_to_set = {'communication':False, 'mental':False}
                features_to_set['communication'] = v.is_sentence_with_communication_verb(raw_sentence)
                features_to_set['mental'] = v.is_sentence_with_mental_verb(raw_sentence)
                features_to_add = [features_to_set[key] for key in sorted(features_to_set.keys())]
                temp.append(features_to_add)

        elif self.featureSetConfiguration == 2:
            v = Verb()
            # connect to db
            for document in X:
                currentSentence = documentCollection.find_one({'document_id': document })
                raw_sentence = currentSentence['raw'].lower()
                features_to_set = {'communication':0, 'mental':0}
                features_to_set['communication'] = v.count_communication_verb_in_sentence(raw_sentence)
                features_to_set['mental'] = v.count_mental_verb_in_sentence(raw_sentence)
                features_to_add = [features_to_set[key] for key in sorted(features_to_set.keys())]
                temp.append(features_to_add)
            min_max_scaler = MinMaxScaler()
            temp = min_max_scaler.fit_transform(temp)

        mongoClient.close()
        features = np.array(temp)
        #print('VerbTypeTransformer:', self.featureSetConfiguration,' ### X:',len(X),'len(features):',len(features))
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return sorted(VerbTypeTransformer.features.keys())
