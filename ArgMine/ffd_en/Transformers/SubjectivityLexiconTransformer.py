from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
import re
from pymongo import MongoClient
parameters = Parameters()
from OPAM.SubjectivityClues import SubjectivityClues

sc = SubjectivityClues()

'''example
featureSetConfiguration 
0 : empty
1 : {'weaksubj-both':Bool,'weaksubj-neutral':Bool,'weaksubj-positive':Bool,'weaksubj-negative':Bool,'strongsubj-neutral':Bool,'strongsubj-both':Bool,'strongsubj-positive':Bool,'strongsubj-negative':Bool}
TODO: 2 : { weaksubj:Bool, strongsubj:Bool,  neutral:Bool, positive:Bool, negative:Bool, both:Bool}
'''

class SubjectivityLexiconTransformer(TransformerMixin, BaseEstimator):
    features = {'weaksubj-both':False, 'weaksubj-neutral':False, 'weaksubj-positive':False, 'weaksubj-negative':False, 'strongsubj-neutral':False, 'strongsubj-both':False, 'strongsubj-positive':False,  'strongsubj-negative':False}

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration
    def transform(self, X, **transform_params):

        # connect to db
        mongoClient = MongoClient('localhost', 27017)
        ffCorpus = mongoClient.FACTFEELCorpus
        # Sentence's table
        documentCollection = ffCorpus.documents
        temp = []


        if self.featureSetConfiguration == 0: # not active
            temp =  [
                [0 for f in sorted(SubjectivityLexiconTransformer.features.keys())]
                for s in X
            ]
        elif self.featureSetConfiguration == 1: # (weaksubj|strongsubj)-(both|neutral|positive|negative)
            for document in X:
                features_to_set = {'weaksubj-both':False, 'weaksubj-neutral':False, 'weaksubj-positive':False, 'weaksubj-negative':False, 'strongsubj-both':False, 'strongsubj-neutral':False, 'strongsubj-positive':False,  'strongsubj-negative':False}
                currentSentence = documentCollection.find_one({'document_id': document })
                raw_sentence = currentSentence['raw'].lower()
                features_in_sentence = sc.analyse_sentence(raw_sentence)
                if features_in_sentence != []:
                    for feat in features_in_sentence:
                        features_to_set[feat] = True
                test = [features_to_set[key] for key in sorted(features_to_set.keys())]
                temp.append(test)
        #TODO : 2
        elif self.featureSetConfiguration == 2:  # weak|strong
            for document in X:
                features_to_set = {'weaksubj':False,'strongsubj':False}
                currentSentence = documentCollection.find_one({'document_id': document })
                raw_sentence = currentSentence['raw'].lower()
                features_in_sentence = sc.analyse_sentence(raw_sentence)
                if features_in_sentence != []:
                    for feat in features_in_sentence:
                        if re.findall('weaksubj',feat) != []:
                            features_to_set['weaksubj'] = True
                        elif re.findall('strongsubj',feat) != []:
                            features_to_set['strongsubj'] = True
                test = [features_to_set[key] for key in sorted(features_to_set.keys())]
                temp.append(test)
        #TODO : 3 (polar | neutral)



        features = np.array(temp)
        #print('SubjectivityLexiconTransformer:' , self.featureSetConfiguration,' ### X:',len(X),'len(features):',len(features))
        mongoClient.close()
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        #TODO: return feature names according to self.featureSetConfiguration
        return sorted(SubjectivityLexiconTransformer.features.keys())
