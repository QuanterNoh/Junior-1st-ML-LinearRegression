import numpy as np

class SimpleLinearRegression:

    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha # learning rate
        self.iterations = iterations
        
        self.coef_ = None 
        self.intercept_ = None 
        self._mse = None # mean squared error

    
    def fit(self, X_data, y_data):
        # W, b initialize
        theta = np.zeros((X_data.shape[1] + 1)) # [b,W] y = 0X + 0
        self.coef_ = theta[1:]
        self.intercept_ = theta[0]

        bias_term = np.ones((X_data.shape[0], 1))
        X = np.concatenate((bias_term, X_data), axis=1)

        # Gradient Descent
        for i in range(self.iterations):
            pred = X @ theta
            error = pred - y_data
            self._mse = np.sum((error ** 2)) / len(X)

            theta -= self.alpha * (X.T @ error) / len(X)
            self.coef_ = theta[1:]
            self.intercept_ = theta[0]
            print(f'Epoch [{i + 1}/{self.iterations}] | MSE: {self._mse:.4f}')

    
    def predict(self, X_data):
        pred = X_data @ self.coef_ + self.intercept_ # hypothesis function
        return pred

    def score(self, X_data, y_data):
        pred = X_data @ self.coef_ + self.intercept_
        
        numerator = ((y_data - pred)**2).sum() # residual sum of squares, RSS
        denominator = ((y_data - y_data.mean())**2).sum() # total sum of squares, SST
        r_square = 1 - (numerator / denominator)
        return r_square

