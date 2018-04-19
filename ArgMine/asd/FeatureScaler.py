"""
FeatureScaler Transformer: Custom Transformer with a set of possible feature scaling techniques 
that can be applied to the dataset to scale or normalize the feature set.
It receives as input a matrix with dimensions X:Y where X corresponds to the total number of 
propositions in the dataset and Y corresponds to the total number of features that were 
generated by previous transformers.
"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfTransformer


class FeatureScaler(TransformerMixin, BaseEstimator):
    
    def __init__(self, scalerType= 0):
        # integer that represents the type of scaler to be applied to the feature set received as input
        self.scalerType= scalerType
    
    def transform(self, X, **transform_params):
        # X -> corresponds to the feature set of the data -> array with the following form: [# samples, # features]
        if self.scalerType == 0:
            return X
        elif self.scalerType == 1:
            return StandardScaler().fit_transform(X)
        elif self.scalerType == 2:
            # Transforms the data to values in the range feature_range=(min, max) (default values: [0, 1])
            return MinMaxScaler().fit_transform(X)
        elif self.scalerType == 3:
            return Normalizer(norm='l2').fit_transform(X)
        elif self.scalerType == 4:
            return TfidfTransformer().fit_transform(X)
        elif self.scalerType == 5:
            # It is meant for data that is already centered at zero or sparse data.
            return MaxAbsScaler().fit_transform(X)
        
    def fit(self, X, y=None, **fit_params):
        return self
        
    def get_feature_names(self):
        return []
    
    def get_content(self):
        return self
    