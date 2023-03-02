import numpy as np
from numpy.linalg import inv

# Task 1

def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    return sum((y_true - y_predicted) ** 2) / y_true.shape[0]
    

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    return 1 - sum((y_true - y_predicted) ** 2) / sum((y_true.mean() - y_true) ** 2)

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here

    def fit(self, X: np.ndarray, y: np.ndarray):
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])
        inverse_matrix = inv(np.transpose(X_c) @ X_c)
        self.weights = inverse_matrix @ np.transpose(X_c) @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])
        return X_c @ self.weights
    
# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.weights = None  # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.l = l

    def fit(self, X: np.ndarray, y: np.ndarray):
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])

        n, m = X_c.shape[0], X_c.shape[1]
        self.weights = np.array([0 for _ in range(m)])

        for _ in range(self.iterations):
            w_grad = np.transpose(X_c) @ (X_c @ self.weights - y) / n + self.l * np.sign(self.weights)
            self.weights = self.weights - self.alpha * w_grad

    def predict(self, X: np.ndarray):
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])
        return X_c @ self.weights


# Task 4

def get_feature_importance(linear_regression):
    return list(abs(linear_regression.weights[1:]))

def get_most_important_features(linear_regression):
    return list(np.argsort(get_feature_importance(linear_regression))[::-1])