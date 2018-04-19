from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from OPAM.ArguingLexicon import ArguingLexicon

al = ArguingLexicon()

"""
featureSetConfiguration

0: empty
1: {'assessment': False, 'authority': False, 'causation': False, 'conditionals': False, 'contrast': False,
                'difficulty': False, 'doubt': False, 'emphasis': False, 'generalization': False, 'inconsistency': False,
                'inyourshoes': False, 'necessity': False, 'possibility': False, 'priority': False,
                'rhetoricalquestion': False, 'structure': False, 'wants': False}
"""


class ArguingLexiconTransformer(TransformerMixin, BaseEstimator):
    features = {'isAssessment': False, 'isAuthority': False, 'isCausation': False, 'isConditional': False,
                                   'isContrast': False,
                                   'isDifficulty': False, 'isDoubt': False, 'isEmphasis': False, 'isGeneralization': False,
                                   'isInconsistency': False,
                                   'isInyourshoes': False, 'isNecessity': False, 'isPossibility': False, 'isPriority': False,
                                   'isRhetoricalQuestion': False, 'isStructure': False, 'isWants': False}
    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):

        mongoClient = MongoClient('localhost', 27017)
        ffCorpus = mongoClient.FACTFEELCorpus
        documentCollection = ffCorpus.documents
        temp = []

        if(self.featureSetConfiguration == 0): # not active
            temp = [
                [0 for f in ArguingLexiconTransformer.features.keys()]
                for s in X
            ]
        elif self.featureSetConfiguration == 1: # active
            for document in X:
                features_to_set = {'isAssessment': False, 'isAuthority': False, 'isCausation': False, 'isConditional': False,
                                   'isContrast': False,
                                   'isDifficulty': False, 'isDoubt': False, 'isEmphasis': False, 'isGeneralization': False,
                                   'isInconsistency': False,
                                   'isInyourshoes': False, 'isNecessity': False, 'isPossibility': False, 'isPriority': False,
                                   'isRhetoricalQuestion': False, 'isStructure': False, 'isWants': False}
                # current learning instance info from database
                currentDocument = documentCollection.find_one({'document_id': document })
                raw_sentence = currentDocument['raw'].lower()
                b =  al.SentenceFragment(raw_sentence)
                for k, l in b.items():
                    if(l != []):
                        features_to_set[k] = True
                #print(len(features_to_set.keys()),features_to_set.keys() )
                temp.append([features_to_set[key] for key in sorted(features_to_set.keys())])

        mongoClient.close()
        features = np.array(temp)
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return sorted(ArguingLexiconTransformer.features.keys())






