from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
from pymongo import MongoClient
parameters = Parameters()
from OPAM.SubjectivityClues import SubjectivityClues

sc = SubjectivityClues()


'''

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
        temp = [
            [0 for f in sorted(SubjectivityLexiconTransformer.features.keys())]
            for s in X
        ]
        if self.featureSetConfiguration == 0: # not active
            pass
        elif self.featureSetConfiguration == 1:
            # connect to db
            mongoClient = MongoClient('localhost', 27017)
            aaecCorpus = mongoClient.AAECCorpus
            # Sentence's table
            sentenceCollection = aaecCorpus.sentence
            temp = []
            for sentenceTuple in X:
                features_to_set = {'weaksubj-both':False, 'weaksubj-neutral':False, 'weaksubj-positive':False, 'weaksubj-negative':False, 'strongsubj-both':False, 'strongsubj-neutral':False, 'strongsubj-positive':False,  'strongsubj-negative':False}
                currentSentence = sentenceCollection.find_one({"$and": [{"articleId": int(sentenceTuple[0])}, {"sentenceId": int(sentenceTuple[1])}]})
                raw_sentence = currentSentence['originalText'].lower()
                features_in_sentence = sc.analyse_sentence(raw_sentence)
                if features_in_sentence != []:
                    for feat in features_in_sentence:
                        features_to_set[feat] = True
                test = [features_to_set[key] for key in sorted(features_to_set.keys())]
                temp.append(test)
        features = np.array(temp)
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return sorted(SubjectivityLexiconTransformer.features.keys())
