from sklearn.base import TransformerMixin, BaseEstimator
from utils.Parameters import Parameters
import numpy as np
parameters = Parameters()


class SubjectivityLexiconTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, featureSetConfiguration = 1 ):
        self.featureSetConfiguration = featureSetConfiguration

    def transform(self, X, **transform_params):
        if self.featureSetConfiguration == 0:
            features = np.array([
                [0 for key in sorted(s.subjectivity_features().keys())]
                for s in X
            ])
        elif self.featureSetConfiguration == 1:
            features = np.array([
                [int(s.subjectivity_features()[key]) for key in sorted(s.subjectivity_features().keys())]
                for s in X
            ])
        return features

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return ['strong','weak']