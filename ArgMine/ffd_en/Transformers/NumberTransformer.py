from sklearn.base import TransformerMixin, BaseEstimator
from nltk.tokenize import word_tokenize
import re
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()


"""
configs:
0 - No feature
1 - numberExists
2 - percentageExists
3 - numberExists & percentageExists
"""


class NumberTransformer(TransformerMixin, BaseEstimator):
    features = {'numberExists': False, 'percentageExists': False}

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        mongoClient = MongoClient('localhost', 27017)
        ffCorpus = mongoClient.FACTFEELCorpus
        temp = []

        # numberExists
        if self.featureSetConfiguration == 1:
            documentCollection = ffCorpus.documents
            for document in X:
                features_to_set = {}
                # current learning instance info from database
                currentDocument= documentCollection.find_one({'document_id': document })
                raw_documet = currentDocument['raw'].lower()
                capture_number = re.findall('\d+',raw_documet)
                if capture_number != [] :
                    features_to_set['numberExists'] = True
                else:
                    features_to_set['numberExists'] = False
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])

        # percentageExists
        elif self.featureSetConfiguration == 2:
            documentCollection = ffCorpus.documents
            for document in X:
                features_to_set = {}
                # current learning instance info from database
                currentDocument = documentCollection.find_one({'document_id': document})
                raw_documet = currentDocument['raw'].lower()
                capture_number = re.findall('%', raw_documet)
                if capture_number != []:
                    features_to_set['percentageExists'] = True
                else:
                    features_to_set['percentageExists'] = False
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])

        # numberExists & percentageExists
        elif self.featureSetConfiguration == 3:
            documentCollection = ffCorpus.documents
            for document in X:
                features_to_set = {}
                # current learning instance info from database
                currentDocument = documentCollection.find_one({'document_id': document})
                raw_documet = currentDocument['raw'].lower()

                capture_number = re.findall('\d+', raw_documet)
                if capture_number != []:
                    features_to_set['numberExists'] = True
                else:
                    features_to_set['numberExists'] = False

                capture_number = re.findall('%', raw_documet)
                if capture_number != []:
                    features_to_set['percentageExists'] = True
                else:
                    features_to_set['percentageExists'] = False
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])

        else :
            temp = [
                [False]
                for s in X
            ]

        mongoClient.close()
        features = np.array(temp)
        return features

    def fit(self, X, y=None, **fit_params):
        return self


    def get_feature_names(self):

        if self.featureSetConfiguration == 1 :
            return ['numberExists']
        elif self.featureSetConfiguration == 2 :
            return ['percentageExists']
        elif self.featureSetConfiguration == 3 :
            return ['numberExists', 'percentageExists']
        else:
            return ['number']






