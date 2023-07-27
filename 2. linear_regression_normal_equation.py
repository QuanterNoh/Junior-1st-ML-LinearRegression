import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X_data, y_data):
        bias_term = np.ones((X_data.shape[0], 1), dtype=float) 
        X = np.concatenate([bias_term, X_data], axis=1)
        
        theta = np.linalg.inv(X.T @ X) @ (X.T @ y_data) # normal_equation
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]


    def predict(self, X_data): 
        pred = X_data @ self.coef_ + self.intercept_
        return pred


    def score(self, X_data, y_data):
        pred = X_data @ self.coef_ + self.intercept_

        numerator = ((y_data - pred)**2).sum()
        denominator = ((y_data - y_data.mean())**2).sum()
        r_square = 1 - (numerator / denominator)
        return r_square