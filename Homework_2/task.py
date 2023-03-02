from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.

        """

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать
            параметры X и y, даже если y не используется).

        """

        dim = X.shape[1]
        n = X.shape[0]
        if self.init == 'random':
            self.centroids = [[-1 for _ in range(dim)] for _ in range(self.n_clusters)]
            for q in range(self.n_clusters):
                for i in range(dim):
                    self.centroids[q][i] = random.uniform(X[:, i].min(), X[:, i].max())

        elif self.init == 'sample':
            self.centroids = [random.choice(X) for _ in range(self.n_clusters)]

        elif self.init == 'k-means++':
            self.centroids = [random.choice(X)]
            while len(self.centroids) != self.n_clusters:
                current = 0
                dist_sqrt_accum = []
                for idx, point in enumerate(X):
                    dist_to_centroid = min([np.linalg.norm(center - point) for center in self.centroids])
                    current += (dist_to_centroid ** 2)
                    dist_sqrt_accum.append(current)

                random_point = random.randint(0, int(dist_sqrt_accum[-1]))
                left, right = -1, n
                while right - left > 1:
                    mid = (left + right) // 2
                    if dist_sqrt_accum[mid] > random_point:
                        right = mid
                    else:
                        left = mid
                self.centroids.append(X[right])

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера,
        к которому относится данный элемент.

        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.

        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров
            (по одному индексу для каждого элемента из X).
        """
        n = X.shape[0]
        dim = X.shape[1]

        labels = [-1 for _ in range(n)]
        centers = copy.deepcopy(self.centroids)
        it = 0
        while it < self.max_iter - 1:
            points_in_centroid = [[] for _ in range(self.n_clusters)]
            for idx, point in enumerate(X):
                label = np.argmin([np.linalg.norm(center - point) for center in centers])
                points_in_centroid[label].append(point)
                labels[idx] = label

            if it == 0:
                flag = False
                for idx, el in enumerate(points_in_centroid):
                    if not el:
                        flag = True

                        if self.init == 'random':
                            for i in range(dim):
                                centers[idx][i] = random.uniform(X[:, i].min(), X[:, i].max())

                        elif self.init == 'sample':
                            centers[idx] = random.choice(X)

                        elif self.init == 'k-means++':
                            centers = list(np.delete(centers, idx, 0))
                            while len(centers) != self.n_clusters:
                                current = 0
                                dist_sqrt_accum = []
                                for idx, point in enumerate(X):
                                    dist_to_centroid = min([np.linalg.norm(center - point) for center in centers])
                                    current += (dist_to_centroid ** 2)
                                    dist_sqrt_accum.append(current)

                                random_point = random.randint(0, int(dist_sqrt_accum[-1]))
                                left, right = -1, n
                                while right - left > 1:
                                    mid = (left + right) // 2
                                    if dist_sqrt_accum[mid] > random_point:
                                        right = mid
                                    else:
                                        left = mid
                                centers.append(X[right])
                if flag:
                    continue

            centers = [np.mean(points, axis=0) for points in points_in_centroid]
            it += 1
        return np.array(labels)


# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.leaf_size = leaf_size

    def dfs(self, adj, v, labels, color, used):
        used[v] = True

        for u in adj[v]:
            if labels[u] == -1:
                labels[u] = color
            if not used[u] and len(adj[u]) >= self.min_samples:
                self.dfs(adj, u, labels, color, used)

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        n = X.shape[0]
        tree = KDTree(X, metric=self.metric, leaf_size=self.leaf_size)
        neighbours = tree.query_radius(X, r=self.eps)
        labels = [-1 for _ in range(n)]
        used = [False for _ in range(n)]
        color = 0

        for i in range(n):
            if not used[i] and len(neighbours[i]) >= self.min_samples:
                labels[i] = color
                self.dfs(neighbours, i, labels, color, used)
                color += 1

        return labels

    
# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """

        self.n_clusters = n_clusters
        self.linkage = linkage

    def get_matrix_dist(self, X):

        n = X.shape[0]
        matrix_dist = [[np.inf for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(X[i] - X[j])
                matrix_dist[i][j] = distance
                matrix_dist[j][i] = distance

        return np.array(matrix_dist)

    def upload_matrix_dist(self, matrix_dist, points_in_claster, i_min, j_min):

        for i in range(matrix_dist.shape[0]):
            if i == i_min:
                matrix_dist[i][i_min] = np.inf
                continue

            elif self.linkage == 'average':
                new_dist = (len(points_in_claster[i_min]) * matrix_dist[i][i_min] + len(points_in_claster[j_min]) *
                            matrix_dist[i][j_min]) / (
                                   len(points_in_claster[i_min]) + len(points_in_claster[j_min]))

            elif self.linkage == 'single':
                new_dist = min(matrix_dist[i][i_min], matrix_dist[i][j_min])

            elif self.linkage == 'complete':
                new_dist = max(matrix_dist[i][i_min], matrix_dist[i][j_min])

            matrix_dist[i][i_min] = new_dist
            matrix_dist[i_min][i] = new_dist

        return matrix_dist

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).
        """

        k = X.shape[0]
        matrix_dist = self.get_matrix_dist(X)
        labels = [i for i in range(k)]
        count = [1 for _ in range(k)]
        points_in_claster = [[i] for i in range(k)]

        while k > self.n_clusters:
            i_min, j_min = np.where(matrix_dist == matrix_dist.min())
            matrix_dist = self.upload_matrix_dist(matrix_dist, points_in_claster, i_min[0], j_min[0])
            matrix_dist = np.delete(matrix_dist, j_min[0], 0)
            matrix_dist = np.delete(matrix_dist, j_min[0], 1)

            points_in_claster[i_min[0]] += points_in_claster[j_min[0]]
            points_in_claster.pop(j_min[0])

            for i in range(1, len(points_in_claster[i_min[0]])):
                el = points_in_claster[i_min[0]][i]
                labels[el] = labels[points_in_claster[i_min[0]][0]]

            k -= 1

        return np.array(labels)
