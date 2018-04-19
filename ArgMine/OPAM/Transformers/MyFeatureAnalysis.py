from sklearn.base import TransformerMixin, BaseEstimator


class MyFeatureAnalysis(TransformerMixin, BaseEstimator):
    def __init__(self, featureAnalysisMethod=0):
        self.featureAnalysisMethod = featureAnalysisMethod

    def transform(self, X, **transform_params):
        # X -> corresponds to the data -> array of propositions

        if self.featureAnalysisMethod == 0:
            return X

    def fit(self, X, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return []

    def get_content(self):
        return self
