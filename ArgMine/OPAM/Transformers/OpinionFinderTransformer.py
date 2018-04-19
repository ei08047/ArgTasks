from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
parameters = Parameters()


class OpinionFinderTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        if self.featureSetConfiguration == 0:
            features = np.array([
                [0]
                for s in X
            ])
        elif self.featureSetConfiguration == 1:
            features = np.array([
                [0 if s.opinion_finder_features()[key] =='subj' else 1 for key in sorted(s.opinion_finder_features().keys())]
                for s in X
            ])
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['is_subj']