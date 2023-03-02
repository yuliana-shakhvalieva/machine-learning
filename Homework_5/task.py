import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F
import random


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass
    
    
class Linear(Module):
    """
    Линейный полносвязный слой.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """

        self.b = np.random.randn(out_features) / ((in_features + out_features) ** (1 / 2))
        self.w = np.random.randn(in_features, out_features) / ((in_features + out_features) ** (1 / 2))

        self.dE_dw = None
        self.dE_db = None
        self.vector = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.dE_dw = x.T

        if len(x.shape) == 1:
            self.vector = True

        return np.array(self.b + x @ self.w)

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """

        if self.vector:
            self.dE_dw = np.outer(self.dE_dw, d)
        else:
            self.dE_dw = self.dE_dw @ d

        self.dE_db = d

        return np.array(d @ self.w.T)

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.w = self.w - alpha * self.dE_dw
        self.b = self.b - alpha * np.sum(self.dE_db, axis=0)


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """

    def __init__(self):
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """

        self.x = x
        return np.maximum(0, x)

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return np.array(d * (self.x >= 0).astype(float))

    

# Task 2
class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и
            описывающий слои нейронной сети.
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.batch_y_pred = None
        self.gradient = None

    def softmax(self, t):
        out = np.exp(t)
        return np.array(out / np.sum(out, axis=1, keepdims=True))

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """

        num_classes = len(set(y))
        n = X.shape[0]
        y_binary = np.eye(num_classes)[y.reshape(1, -1)[0].astype(int)]
        index = [i for i in range(n)]

        for epoch in range(self.epochs):
            shuffle_index = np.random.permutation(index)

            for i in range(n // self.batch_size):
                mask = shuffle_index[i * self.batch_size: i * self.batch_size + self.batch_size]
                self.batch_y_pred, batch_y = X[mask], y_binary[mask]

                # Forward
                for module in self.modules:
                    self.batch_y_pred = module.forward(self.batch_y_pred)

                z = self.softmax(self.batch_y_pred)

                # Backward
                self.gradient = z - batch_y

                for module in self.modules[::-1]:
                    self.gradient = module.backward(self.gradient)

                # Update
                for module in self.modules:
                    module.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        y_pred = X
        for module in self.modules:
            y_pred = module.forward(y_pred)

        return self.softmax(y_pred)

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


    
# Task 3

classifier_moons = MLPClassifier(modules=[Linear(2, 4), ReLU(), Linear(4, 2)], epochs=100, alpha=0.001, batch_size=100) # Нужно указать гиперпараметры
classifier_blobs = MLPClassifier(modules=[Linear(2, 4), ReLU(), Linear(4, 3)], epochs=100, alpha=0.001, batch_size=100) # Нужно указать гиперпараметры


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(6, 10, 3, padding=1, stride=2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(640, 300)
        self.linear_2 = nn.Linear(300, 100)
        self.linear_3 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return self.softmax(x)

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] +"model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        torch.load(__file__[:-7] + "model.pth")

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        torch.save(self.state_dict(), 'weights')



def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    loss = nn.CrossEntropyLoss()
    return loss(model(X), y)
    # y_binary = torch.eye(10)[y]
    # return -torch.mean(y_binary * torch.log(model(X.cuda())), 1)[0]