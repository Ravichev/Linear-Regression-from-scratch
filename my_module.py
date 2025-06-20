import numpy as np

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, num_iter=1000, epsilon=1e-10):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.weights = None
        self.loss_history = []

    def normalize_features(self, X):
        """Returning normalized data from features of X."""
        mean_values = np.mean(X, axis=0)
        std_values = np.std(X, axis=0)
        X_normalized = (X - mean_values) / std_values
        return X_normalized

    def add_intercept(self, X):
        """Add row of ones to X matrix (adding intercept)/"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        """Training linear regression model."""
        X = self.add_intercept(X)
        m, n = X.shape
        self.weights = np.zeros(n)

        for epoch in range(self.num_iter):
            y_pred = X.dot(self.weights)
            loss = np.sum((y_pred - y) ** 2) / m  #2 m
            self.loss_history.append(loss)

            gradient = 2/m * X.T.dot(y_pred - y)
            self.weights -= self.learning_rate * gradient

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{self.num_iter}, MSE: {loss}')
            if loss < self.epsilon:
                print(f'Converged. Epoch {epoch + 1}, Final MSE: {loss}')
                break

        return self

    def predict(self, X):
        """Predicting values for the new data based on weigths."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        if isinstance(X, list):
            X = np.array(X)
        X = self.add_intercept(X)
        if X.shape[1] != len(self.weights):
            raise IndexError(f'X shape({X.shape[1]}) not equal the number of weights ({len(self.weights)})')
        return X.dot(self.weights)

    def get_loss_history(self):
        """Returning loss history"""
        return self.loss_history

    def get_weights(self):
        """Returning model weights"""
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        return self.weights
