from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class DenseTransformer(TransformerMixin, BaseEstimator):

    def transform(self, X, y=None, **fit_params):
        if isinstance(X, np.ndarray):
            return X
        else:
            return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        return []
    
    def get_content(self):
        return self