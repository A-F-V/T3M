from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapper):
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vectorize(self.mapper)(X)  # <--- self.key
