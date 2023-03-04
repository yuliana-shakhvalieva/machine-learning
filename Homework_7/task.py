import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False


# Task 1

class LinearSVM:
    def __init__(self, C: float):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.

        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        n, m = X.shape
        kernel = np.array([X @ el.T for el in X])

        y_matrix = y.reshape(1, -1)
        P = matrix(y_matrix.T @ y_matrix * kernel)
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        A = matrix(y_matrix.astype('float'))
        q = matrix(-np.ones((n, 1)))
        h = matrix(np.vstack((np.zeros((n, 1)), np.ones((n, 1)) * self.C)))
        b = matrix(np.zeros(1))

        svm_parameters = solvers.qp(P, q, G, h, A, b)

        alphas = np.array(svm_parameters['x'])[:, 0]
        threshold = 1e-5
        self.support = (self.alphas > threshold).reshape(-1, )
        self.w = alphas * X.T @ y
        self.b = np.mean(y[self.support] - X[self.support] @ self.w)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        return X @ self.w.T + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))


# Task 2

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"

    def polynomial_kernel(X, y):
        return (X @ y.T + c) ** power

    return polynomial_kernel


def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"

    def gaussian_kernel(X, y):
        return np.exp(-sigma * np.linalg.norm(X - y, axis=1) ** 2)

    return gaussian_kernel


# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.

        """
        self.C = C
        self.kernel = kernel
        self.support = None
        self.b = None
        self.alphas = None
        self.y = None
        self.X = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        n, m = X.shape
        self.y = y
        self.X = X
        kernel = np.array([self.kernel(X, el) for el in X])
        y_matrix = y.reshape(1, -1)
        P = matrix(y_matrix.T @ y_matrix * kernel)
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        A = matrix(y_matrix.astype('float'))
        q = matrix(-np.ones((n, 1)))
        h = matrix(np.vstack((np.zeros((n, 1)), np.ones((n, 1)) * self.C)))
        b = matrix(np.zeros(1))

        svm_parameters = solvers.qp(P, q, G, h, A, b)

        self.alphas = np.array(svm_parameters['x'])[:, 0]
        threshold = 1e-5
        self.support = (self.alphas > threshold).reshape(-1, )
        self.b = np.mean(
            y[self.support] - np.sum((self.alphas.reshape(-1, 1) * y.reshape(-1, 1) * kernel), axis=0)[self.support])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        kernel = np.array([self.kernel(self.X, el) for el in X])

        return np.sum(self.alphas.reshape(-1, 1) * self.y.reshape(-1, 1) * kernel.T, axis=0) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))