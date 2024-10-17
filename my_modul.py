import numpy as np

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, num_iter=1000, epsilon=1e-10):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.weights = None
        self.loss_history = []

    def normalize_features(self, X):
        """Нормализует признаки X, возвращает нормализованные данные."""
        mean_values = np.mean(X, axis=0)
        std_values = np.std(X, axis=0)
        X_normalized = (X - mean_values) / std_values
        return X_normalized

    def add_intercept(self, X):
        """Добавляет столбец единиц к матрице X для интерсепта."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        """Тренирует модель линейной регрессии."""
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
        """Предсказывает значения для новых данных на основе обученных весов."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        if isinstance(X, list):
            X = np.array(X)
        X = self.add_intercept(X)
        if X.shape[1] != len(self.weights):
            raise IndexError(f'X shape({X.shape[1]}) not equal the number of weights ({len(self.weights)})')
        return X.dot(self.weights)

    def get_loss_history(self):
        """Возвращает историю потерь."""
        return self.loss_history

    def get_weights(self):
        """Возвращает веса модели."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet!")
        return self.weights