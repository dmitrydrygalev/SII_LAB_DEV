import numpy as np
import pandas as pd

#Функция по разделению исходно датафрейма на тестовую и тренировочную выборки-------------------------------------------------------
def trainTestSpliter(x, y, test_size, random_seed=0):

    np.random.seed(random_seed) #устанавливает начальное значение для генератора случайных чисел

    num_train = int(len(x) * (1 - test_size)) #получение номера идекса для разделения датасета

    dataset = pd.concat([x, y], axis=1) #объединение двух датасетов (x и y) по столбцам в результате получается один датасет содержащий данные из обоих исходных датасетов, расположенные друг под другом

    indices = np.array(dataset.index) #превращает индексы датасета dataset в массив numpy

    np.random.shuffle(indices) #перемешивает элементы в массиве indices

    train_dataset = pd.DataFrame([dataset.loc[i, :] for i in indices[:num_train]]) #train_dataset содержит выборки из dataset, где индексом выборки является элемент из массива indices, взятый до num_train
    test_dataset = pd.DataFrame([dataset.loc[i, :] for i in indices[num_train:]]) #test_dataset содержит выборки из dataset, где индексом выборки является элемент из массива indices, взятый после num_train

    x_train = train_dataset.iloc[:, :-1] #выбирает все строки и все столбцы из train_dataset, исключая последнее значение (target)
    y_train = pd.DataFrame(train_dataset.iloc[:, -1].to_numpy().reshape(x_train.shape[0], 1)) #датафрейм с одной колонкой, содержащей target значения

    x_test = test_dataset.iloc[:, :-1] #выбирает все строки и все столбцы из test_dataset, исключая последнее значение (target)
    y_test = pd.DataFrame(test_dataset.iloc[:, -1].to_numpy().reshape(x_test.shape[0], 1)) #датафрейм с одной колонкой, содержащей target значения

    return x_train, y_train, x_test, y_test
#Функция по разделению исходно датафрейма на тестовую и тренировочную выборки-------------------------------------------------------

