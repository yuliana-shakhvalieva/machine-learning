from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List


# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    unique, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return np.sum(p * (1 - p))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    unique, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return - np.sum(p * np.log2(p))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """

    len_l = left_y.shape[0]
    len_r = right_y.shape[0]
    return criterion(np.concatenate((left_y, right_y))) - \
           len_l * criterion(left_y) / (len_l + len_r) - \
           len_r * criterion(right_y) / (len_l + len_r)


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        values, counts = np.unique(ys, return_counts=True)
        ind = np.argmax(counts)
        self.y = values[ind]
        self.prob = {value: count / len(ys) for value, count in zip(values, counts)}


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


# Task 3

def right_is_empty(best_left_mask):
    return all(best_left_mask)


def left_is_empty(best_left_mask):
    return not any(best_left_mask)


def count_min_leaf(best_left_mask):
    return min(sum(best_left_mask), sum((~best_left_mask)))


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        if criterion == 'gini':
            self.criterion = gini
        elif criterion == 'entropy':
            self.criterion = entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.root = self.built_tree(X, y, 0)

    def built_tree(self, X, y, level):
        best_left_mask = []
        split_dim = split_value = best_gain = 0
        for i in range(X.shape[1]):
            unique = np.unique(X[:, i])
            for j in range(0, len(unique), 3):
                value = unique[j]
                left_mask = X[:, i] < value
                current_gain = gain(y[left_mask], y[~left_mask], self.criterion)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_left_mask = left_mask
                    split_dim = i
                    split_value = value
                    
        level += 1
        if right_is_empty(best_left_mask) or \
            left_is_empty(best_left_mask) or \
            level == self.max_depth or \
            count_min_leaf(best_left_mask) <= self.min_samples_leaf:
            return DecisionTreeLeaf(y)

        x_left, y_left = X[best_left_mask], y[best_left_mask]
        x_right, y_right = X[~best_left_mask], y[~best_left_mask]

        return DecisionTreeNode(split_dim=split_dim, split_value=split_value,
                                left=self.built_tree(x_left, y_left, level),
                                right=self.built_tree(x_right, y_right, level))

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        return [self.get_label(self.root, el) for el in X]

    def get_label(self, node, X):
        if isinstance(node, DecisionTreeLeaf):
            return node.prob

        if X[node.split_dim] < node.split_value:
            return self.get_label(node.left, X)
        else:
            return self.get_label(node.right, X)

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


# Task 4
task4_dtc = DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=4)

