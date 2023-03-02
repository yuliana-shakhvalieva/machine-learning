import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn
from sklearn.metrics import accuracy_score
import math


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = None
        self.iterations = iterations
        self.true_labels = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон.
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """

        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])

        n, m = X_c.shape[0], X_c.shape[1]
        self.w = np.array([0 for _ in range(m)])

        self.true_labels = list(set(y))
        y_norm = np.array([0] * n)
        y_norm[y == self.true_labels[0]] = 1
        y_norm[y == self.true_labels[1]] = -1

        for _ in range(self.iterations):
            h = np.sign(X_c @ self.w)
            h[h == 0] = 1
            mask = (y_norm != h)
            self.w = self.w + np.sum(np.multiply(y_norm[mask][np.newaxis, :].T, X_c[mask]), axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])

        y = np.sign(X_c @ self.w)
        labels = np.array([0] * X_c.shape[0])

        labels[y == 1] = self.true_labels[0]
        labels[y == -1] = self.true_labels[1]

        return np.array(labels)
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = None
        self.iterations = iterations
        self.true_labels = None
        self.w_best = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса,
        при которых значение accuracy было наибольшим.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])

        n, m = X_c.shape[0], X_c.shape[1]
        self.w = np.array([0] * m)

        self.true_labels = list(set(y))
        y_norm = np.array([0] * n)
        y_norm[y == self.true_labels[0]] = 1
        y_norm[y == self.true_labels[1]] = -1

        best_score = 1

        for _ in range(self.iterations):
            h = np.sign(X_c @ self.w)
            mask = (y_norm != h)
            score = np.sum(mask) / n
            if score < best_score:
                best_score = score
                self.w_best = np.copy(self.w)

            self.w = self.w + np.sum(np.multiply(y_norm[mask][np.newaxis, :].T, X_c[mask]), axis=0)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        constant = np.array([[1] for _ in range(X.shape[0])])
        X_c = np.hstack([constant, X])

        y = np.sign(X_c @ self.w_best)

        labels = np.array([0] * X_c.shape[0])
        labels[y == 1] = self.true_labels[0]
        labels[y == -1] = self.true_labels[1]

        return np.array(labels)
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.

    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    result = []
    image_height = images.shape[1]
    image_width = images.shape[2]

    for image in images:
        symmetry = (abs(image[:image_height//2, :] - image[::-1, ::][:image_height//2, :]) > 0.3).sum()
        intensity = np.sum(image)
        result.append(np.array([symmetry, intensity]))

    return np.array(result)
