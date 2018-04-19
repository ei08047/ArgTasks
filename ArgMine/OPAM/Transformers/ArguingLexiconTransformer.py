from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
parameters = Parameters()


class ArguingLexiconTransformer(TransformerMixin, BaseEstimator):
    features = {'assessment': False, 'authority': False, 'causation': False, 'conditionals': False, 'contrast': False,
                'difficulty': False, 'doubt': False, 'emphasis': False, 'generalization': False, 'inconsistency': False,
                'inyourshoes': False, 'necessity': False, 'possibility': False, 'priority': False,
                'rhetoricalquestion': False, 'structure': False, 'wants': False}

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        if(self.featureSetConfiguration == 0): # not active
            features = np.array([
                ArguingLexiconTransformer.features.keys()
                for s in X
            ])
        elif self.featureSetConfiguration == 1: # active
            features = np.array([
                [s.arguing_features()[key] for key in sorted(s.arguing_features().keys())]
                for s in X
            ])
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return sorted(ArguingLexiconTransformer.features.keys())






