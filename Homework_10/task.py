import numpy as np
import pandas
import random
import copy
import math
from typing import NoReturn

# Task 1

def cyclic_distance(points, dist):
    distance = 0
    for i in range(len(points)-1):
        distance += dist(points[i], points[i+1])
    distance += dist(points[0], points[-1])
    return distance


def l2_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def l1_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist  # Do not change

    def optimize(self, X):
        return self.optimize_explain(X)[-1]

    def optimize_explain(self, X):
        best_index = list(range(X.shape[0]))
        best_dist = cyclic_distance(X, self.dist)
        answer = [best_index]

        for it in range(self.max_iterations):
            found_better = False
            for i in range(X.shape[0] - 1):
                for j in range(i+1, X.shape[0]):
                    new_index = copy.copy(best_index)
                    new_index[i], new_index[j] = new_index[j], new_index[i]
                    current_dist = cyclic_distance(X[new_index], self.dist)
                    if current_dist < best_dist:
                        found_better = True
                        best_dist = current_dist
                        best = copy.copy(new_index)

            if not found_better:
                return answer

            best_index = copy.copy(best)
            answer.append(copy.copy(best_index))

        return answer


    
# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
        self.n = None
        self.individual = None
        self.mutate_prob = 0.7
        self.best_distance = math.inf
        self.found_better = False

    def optimize(self, X):
        self.n = X.shape[0]
        self.optimize_explain(X)
        return self.individual

    def optimize_explain(self, X):
        index = list(range(X.shape[0]))
        population = [np.random.permutation(index) for _ in range(self.pop_size)]

        for it in range(self.iters):
            self.found_better = False
            survivors = self.get_best_individuals(X, population)
            if not self.found_better:
                return
            population = self.get_new_population(survivors)

    def get_new_population(self, survivors):
        new_population = [survivor for survivor in survivors]

        for i in range(self.pop_size - self.surv_size):
            individual_1_idx, individual_2_idx = np.random.choice(range(self.surv_size), 2)
            individual_1, individual_2 = survivors[individual_1_idx], survivors[individual_2_idx]
            new_individual = self.crossing_over(individual_1, individual_2)
            new_population.append(new_individual)

        return new_population

    def crossing_over(self, individual_1, individual_2):
        length = random.randint(2, self.n)
        l = random.randint(0, self.n - length)
        r = l + length
        new_individual = individual_1[l: r]
        unique_value = set(new_individual)

        j = 0
        while new_individual.shape[0] != self.n:
            if individual_2[j] not in unique_value:
                new_individual = np.append(new_individual, individual_2[j])
            j += 1

        new_individual = self.mutate(new_individual)

        return new_individual

    def mutate(self, individual):
        if random.random() < self.mutate_prob:
            i = random.randint(0, self.n - 2)
            j = random.choice([i+1, i-1])
            individual[i], individual[j] = individual[j], individual[i]

        return individual

    def get_best_individuals(self, X, population):
        distances = []
        for i in range(self.pop_size):
            individual = population[i]
            distance = cyclic_distance(X[individual], self.dist)
            distances.append((distance, i))
            if distance < self.best_distance:
                self.best_distance = distance
                self.found_better = True
                self.individual = individual

        best_distances = sorted(distances, key=lambda x: x[0])[:self.surv_size]
        return np.array([population[idx[1]] for idx in best_distances])


# Task 4

class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        values, counts = np.unique(' '.join(X).split(), return_counts=True)
        self.dictionary = []
        for value, count in zip(values, counts):
            self.dictionary.append((value, count))

        self.dictionary = sorted(self.dictionary, reverse=True, key=lambda x: x[1])

        if len(self.dictionary) > voc_limit:
            self.dictionary = self.dictionary[:voc_limit + 1]

        self.voc_limit = voc_limit

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            который необходимо векторизовать.

        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        matrix = []
        for el in X:
            vector = np.zeros(self.voc_limit)
            words = list(map(str, el.split()))
            for i in range(self.voc_limit):
                if self.dictionary[i][0] in words:
                    vector[i] = 1
            matrix.append(vector)

        return np.array(matrix)

# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """
        pass
        
    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]
    
    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу. 
            Матрица размера (X.shape[0], n_classes)
        """
        return None