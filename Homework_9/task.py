from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 0

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


# Task 1

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


def right_is_empty(best_left_mask):
    return all(best_left_mask)


def left_is_empty(best_left_mask):
    return not any(best_left_mask)


def count_min_leaf(best_left_mask):
    return min(sum(best_left_mask), sum((~best_left_mask)))


class DecisionTree:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
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
        n = X.shape[0]
        list_range = range(n)

        bag_idx = np.random.choice(list_range, size=n, replace=True)
        out_bag_idx = np.array(list(set(list_range) - set(bag_idx)))

        self.X_bag = X[bag_idx]
        self.y_bag = y[bag_idx]

        self.X_out_bag = X[out_bag_idx]
        self.y_out_bag = y[out_bag_idx]

        if criterion == 'gini':
            self.criterion = gini
        elif criterion == 'entropy':
            self.criterion = entropy

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        if max_features == 'auto':
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            self.max_features = max_features

        self.root = self.built_tree(self.X_bag, self.y_bag, 0)

    def built_tree(self, X, y, depth):
        best_left_mask = []
        split_dim = split_value = best_gain = 0

        features_idx = np.random.choice(range(X.shape[1]), size=self.max_features, replace=False)
        for i in features_idx:
            value = X[0, i]
            left_mask = X[:, i] == value
            current_gain = gain(y[left_mask], y[~left_mask], self.criterion)
            if current_gain > best_gain:
                best_gain = current_gain
                best_left_mask = left_mask
                split_dim = i
                split_value = value

        depth += 1
        if right_is_empty(best_left_mask) or \
                left_is_empty(best_left_mask) or \
                depth == self.max_depth or \
                count_min_leaf(best_left_mask) <= self.min_samples_leaf:
            return DecisionTreeLeaf(y)

        x_left, y_left = X[best_left_mask], y[best_left_mask]
        x_right, y_right = X[~best_left_mask], y[~best_left_mask]

        return DecisionTreeNode(split_dim=split_dim, split_value=split_value,
                                left=self.built_tree(x_left, y_left, depth),
                                right=self.built_tree(x_right, y_right, depth))

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

        if X[node.split_dim] == node.split_value:
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
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.random_forest = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            self.random_forest.append(DecisionTree(X, y,
                                                   criterion=self.criterion,
                                                   max_depth=self.max_depth,
                                                   min_samples_leaf=self.min_samples_leaf,
                                                   max_features=self.max_features))

    def predict(self, X):
        predicted_matrix = []
        for tree in self.random_forest:
            predicted_matrix.append(tree.predict(X))

        predicted_matrix = np.array(predicted_matrix)
        unique_values, indexes = np.unique(predicted_matrix, return_inverse=True)
        return unique_values[
            np.argmax(
                np.apply_along_axis(np.bincount, 0, indexes.reshape(predicted_matrix.shape), None, np.max(indexes) + 1),
                axis=0)]

    
# Task 3

def count_accuracy(y, y_pred):
    return np.mean(y == y_pred)


def tree_feature_importance(tree):
    X = tree.X_out_bag
    y = tree.y_out_bag
    y_pred = tree.predict(X)
    accuracy = count_accuracy(y, y_pred)
    importance = []

    for i in range(X.shape[1]):
        X_shuffle = copy.deepcopy(X)
        np.random.shuffle(X_shuffle[:, i])

        y_pred_shuffle = tree.predict(X_shuffle)
        accuracy_shuffle = count_accuracy(y, y_pred_shuffle)
        importance.append(accuracy - accuracy_shuffle)

    return np.array(importance)


def feature_importance(rfc):
    importance_matrix = []
    for tree in rfc.random_forest:
        importance_matrix.append(tree_feature_importance(tree))

    return np.mean(np.array(importance_matrix), axis=0)

# Task 4

rfc_age = RandomForestClassifier(criterion="gini", max_depth=14, min_samples_leaf=5, max_features="auto", n_estimators=11)
rfc_gender = RandomForestClassifier(criterion="gini", max_depth=10, min_samples_leaf=3, max_features="auto", n_estimators=10)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model

catboost_rfc_age = CatBoostClassifier(loss_function='MultiClass',
                                      iterations=10,
                                      learning_rate=1,
                                      depth=2)
catboost_rfc_age.load_model(__file__[:-7] + 'catboost_age.cbm', format='cbm')

catboost_rfc_gender = CatBoostClassifier(loss_function='MultiClass',
                                         iterations=10,
                                         learning_rate=1,
                                         depth=2)
catboost_rfc_gender.load_model(__file__[:-7] + 'catboost_gender.cbm', format='cbm')