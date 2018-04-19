from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
parameters = Parameters()


class VerbTypeTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        features = np.array([
            [s.verb_features()[key] for key in sorted(s.verb_features().keys())]
            for s in X
        ])
        return features

    def fit(self, X, y=None, **fit_params):
        return self


    def get_feature_names(self):
        return ['communication','mental']
