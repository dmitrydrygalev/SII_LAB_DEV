import pandas as pd
import matplotlib.pyplot as plt
from MR import *
from utils import *

#Чтение файла-------------------------------------------------------
data = pd.read_csv("./data/housing.csv")
#Чтение файла-------------------------------------------------------


#Разделение датасета на выборку результирующего признака и выборку всех остальных признаков-----------------------------------------
dataset = data.copy() #копирование дата сета
Y = dataset['Y'] #выборка только результирующего признака
x = dataset.drop('Y', axis = 1) #выборка всех признаков кроме результирующего
x_train, y_train, x_test, y_test = trainTestSpliter(x, Y, 0.2, 123) #получение тестовых и тренировочных выборок
#Разделение датасета на выборку результирующего признака и выборку всех остальных признаков-----------------------------------------


train_scores, test_scores = [], [] #тренировочный и тестовый массивы для записи ошибок
reg_regr = MultipleRegression() #инициализация класса множественной регрессии


for l in np.logspace(-5, 10, 10): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10
    print(f"Метрика L составляет - {l}")

    steps, errors = reg_regr.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="ElasticNet", lam=l, rho=0.1)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    train_scores.append(reg_regr.MSE(x_i, y_train)) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    test_scores.append(reg_regr.MSE(x_i, y_test)) #добавление значения средней квадратичной ошибки в массив для тестовой выборки

plt.plot(train_scores, 'g') #построение графика по массиву средних квадратичных ошибок для тренировочной выборки
plt.plot(test_scores, 'r') #построение графика по массиву средних квадратичных ошибок для тестовой выборки