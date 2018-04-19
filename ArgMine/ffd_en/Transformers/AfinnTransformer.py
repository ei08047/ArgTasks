from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from afinn import Afinn


afinn = Afinn()

"""
configs:
0- No feature
1- sentence polarity TODO:(needs parameterization )
2- num_polarity_words / num_neutral_words
3- mean_pos mean_neg
4- median_pos median_neg
"""


class AfinnTransformer(TransformerMixin, BaseEstimator):
    features = {'afinn_value': 0, 'polar_neutral_ratio': 0  }

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):

        mongoClient = MongoClient('localhost', 27017)
        ffCorpus = mongoClient.FACTFEELCorpus
        temp = [
            [0 for f in sorted(AfinnTransformer.features.keys())]
            for s in X
        ]
        # document_score
        if self.featureSetConfiguration == 1:
            documentCollection = ffCorpus.documents
            temp = []
            for document in X:
                features_to_set = AfinnTransformer.features
                # current learning instance info from database
                currentDocument= documentCollection.find_one({'document_id': document })
                raw_documet = currentDocument['raw'].lower()
                b = afinn.score(raw_documet)
                features_to_set['afinn_value'] = b

                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])
            min_max_scaler = MinMaxScaler()
            temp = min_max_scaler.fit_transform(temp)
        # num_polarity_words / num_neutral_words
        elif self.featureSetConfiguration == 2:
            documentCollection = ffCorpus.documents
            temp = []
            for document in X:
                features_to_set = AfinnTransformer.features
                # current learning instance info from database
                currentDocument = documentCollection.find_one({'document_id': document})
                raw_document = currentDocument['raw'].lower()

                words = word_tokenize(raw_document)
                scores_list = [afinn.score(word) for word in words]

                neutral = [zero for zero in scores_list if zero==0]
                polar = [pol for pol in scores_list if pol != 0]

                if(neutral !=[] and polar!=[]):
                    b = len(polar)/len(neutral)
                else:
                    b = 0
                features_to_set['polar_neutral_ratio'] = b
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])
        # mean_pos mean_neg
        elif self.featureSetConfiguration == 3:
            documentCollection = ffCorpus.documents
            temp = []
            for document in X:
                features_to_set = AfinnTransformer.features
                # current learning instance info from database
                currentDocument= documentCollection.find_one({'document_id': document })
                raw_document = currentDocument['raw'].lower()

                words = word_tokenize(raw_document)
                scores_list = [afinn.score(word) for word in words]

                neutral = [zero for zero in scores_list if zero==0]
                polar = [pol for pol in scores_list if pol != 0]

                if(neutral !=[] and polar!=[]):
                    b_r = len(polar)/len(neutral)
                else:
                    b_r=0

                b_a = afinn.score(raw_document)
                features_to_set['afinn_value'] = b_a
                features_to_set['polar_neutral_ratio'] = b_r
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])

            min_max_scaler = MinMaxScaler()
            temp = min_max_scaler.fit_transform(temp)
        # median_pos median_neg
        elif self.featureSetConfiguration == 4:
            documentCollection = ffCorpus.documents
            temp = []
            for document in X:
                features_to_set = {'polar/neutral': 0}

        mongoClient.close()
        features = np.array(temp)
        #print('AfinnTransformer:' , self.featureSetConfiguration,' ### X:',len(X),'len(features):',len(features))
        return features

    def fit(self, X, y=None, **fit_params):
        return  self

    ## names are related to featureSetConfiguration
    def get_feature_names(self):
        return sorted(AfinnTransformer.features.keys())






