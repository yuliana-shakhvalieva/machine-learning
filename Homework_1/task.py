import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:

    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    
    df = pd.read_csv(path_to_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['label']
    X = df.drop('label', axis=1)
    y = y.replace('M', 1)
    y = y.replace('B', 0)
    
    return np.array(X), np.array(y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    df = pd.read_csv(path_to_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['label']
    X = df.drop('label', axis=1)

    return np.array(X), np.array(y)

    
# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    size = X.shape[0]
    delimiter = round(size * ratio)
    return X[:delimiter], y[:delimiter], X[delimiter:], y[delimiter:]

    
# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    
    classes = list(set(y_true))
    
    n = len(classes)
    TP = [0 for _ in range(n)]
    FP = [0 for _ in range(n)]
    FN = [0 for _ in range(n)]
    TN = [len(y_true) for _ in range(n)]

    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            TP[true] += 1
            TN[true] -= 1
            correct += 1
        else:
            FN[true] += 1
            TN[true] -= 1

            FP[pred] += 1
            TN[pred] -= 1


    precision = [-1 for _ in range(n)]
    recall = [-1 for _ in range(n)]

    for clazz in classes:
        precision[clazz] = TP[clazz] / (TP[clazz] + FP[clazz])
        recall[clazz] = TP[clazz] / (TP[clazz] + FN[clazz])

    return np.array(precision), np.array(recall),  correct / len(y_true)

    
# Task 4

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.leaf_size = leaf_size
        self.idx = np.arange(len(X))
        self.X_idx = np.array(list(zip(X, self.idx)), dtype=object)
        self.tree = self.build_tree(self.X_idx)

    def build_tree(self, points_idx, depth=0):
        n = points_idx.shape[0]

        if n > self.leaf_size:

            axis = depth % points_idx.shape[1]

            sorted_points_idx = sorted(points_idx, key=lambda point: point[0][axis])

            return {'point': sorted_points_idx[n // 2],
                    'left': self.build_tree(np.array(sorted_points_idx[:n // 2]), depth + 1),
                    'right': self.build_tree(np.array(sorted_points_idx[n // 2 + 1:]), depth + 1)}

        elif n >= 1:

            return {'point': points_idx,
                    'left': None,
                    'right': None}

    def merge(self, left, right):
        len_left, len_right = len(left), len(right)
        result = [0 for _ in range(len_left + len_right)]
        pA = pB = pC = 0

        while pA != len_left and pB != len_right:

            if left[pA][0] <= right[pB][0]:
                result[pC] = left[pA]
                pC += 1
                pA += 1
            else:
                result[pC] = right[pB]
                pC += 1
                pB += 1

        while pA != len_left:
            result[pC] = left[pA]
            pC += 1
            pA += 1

        while pB != len_right:
            result[pC] = right[pB]
            pC += 1
            pB += 1

        return result

    def compare_neighbours(self, distance_1, distance_2, k):
        if distance_1 is not None:
            distance_1 = sorted(distance_1, key=lambda x: x[0])
        else:
            distance_1 = []

        if distance_2 is not None:
            distance_2 = sorted(distance_2, key=lambda x: x[0])
        else:
            distance_2 = []

        return self.merge(distance_1, distance_2)[:k]

    def get_neighbours(self, tree, point_idx, k, depth=0):
        if tree is None:
            return None

        axis = depth % len(point_idx)

        if tree.get('left') is not None:

            if point_idx[0][axis] < tree.get('point')[0][axis]:
                next_tree = tree.get('left')
                opposite_tree = tree.get('right')
            else:
                next_tree = tree.get('right')
                opposite_tree = tree.get('left')

            leaf_distance = [[np.linalg.norm(point_idx[0] - tree.get('point')[0]), tree.get('point')]]

        else:
            if point_idx[0][axis] < tree.get('point')[0][0][axis]:
                next_tree = tree.get('left')
                opposite_tree = tree.get('right')
            else:
                next_tree = tree.get('right')
                opposite_tree = tree.get('left')

            leaf_distance = [[np.linalg.norm(point_idx[0] - node_el[0]), node_el] for node_el in tree.get('point')]

        distance_neighbours = self.compare_neighbours(self.get_neighbours(next_tree, point_idx, k, depth + 1),
                                                      leaf_distance, k)

        min_distance = distance_neighbours[-1][0]

        if tree.get('left') is not None:
            if min_distance > point_idx[0][axis] - tree.get('point')[0][axis]:
                distance_neighbours = self.compare_neighbours(
                    self.get_neighbours(opposite_tree, point_idx, k, depth + 1),
                    distance_neighbours, k)
        else:
            if min_distance > point_idx[0][axis] - tree.get('point')[0][0][axis]:
                distance_neighbours = self.compare_neighbours(
                    self.get_neighbours(opposite_tree, point_idx, k, depth + 1),
                    distance_neighbours, k)

        return distance_neighbours

    def query(self, X: np.array, k: int = 1):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        ans = [[] for _ in range(len(X))]
        for idx, point in enumerate(X):

            for el in self.get_neighbours(self.tree, (point, idx), k):
                ans[idx].append(el[1][1])

        return ans        

        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array):
        self.tree = KDTree(X, self.leaf_size)
        self.y = y
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """

    def predict_proba(self, X: np.array):

        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """

        neighbors = self.tree.query(X, self.n_neighbors)
        count_clazz = np.unique(self.y).shape[0]
        ans = [[0 for _ in range(count_clazz)] for _ in range(X.shape[0])]

        for idx, neighbor in enumerate(neighbors):
            for el in neighbor:
                clazz = self.y[el]
                ans[idx][clazz] += 1

        for idx, el in enumerate(ans):
            ans[idx] = [x / count_clazz for x in el]

        return ans

    def predict(self, X: np.array) -> np.array:

        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(self.predict_proba(X), axis=1)