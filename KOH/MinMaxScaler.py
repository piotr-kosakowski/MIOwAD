import numpy as np
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
    
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
    
    def transform(self, X):
        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        scaled_range = self.feature_range[1] - self.feature_range[0]
        X_scaled = X_scaled * scaled_range + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        scaled_range = self.feature_range[1] - self.feature_range[0]
        X = (X_scaled - self.feature_range[0]) / scaled_range
        X = X * (self.max_ - self.min_) + self.min_
        return X