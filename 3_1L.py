import matplotlib.pyplot as plt
from MR import *
from utils import *
import pandas as pd

#Чтение файла-------------------------------------------------------
data = pd.read_csv("./data/housing.csv")
#Чтение файла-------------------------------------------------------

#Разделение датасета на выборку результирующего признака и выборку всех остальных признаков-----------------------------------------
dataset = data.copy() #копирование дата сета
Y = dataset['Y'] #выборка только результирующего признака
x = dataset.drop('Y', axis = 1) #выборка всех признаков кроме результирующего
x_train, y_train, x_test, y_test = trainTestSpliter(x, Y, 0.2, 123) #получение тестовых и тренировочных выборок
#Разделение датасета на выборку результирующего признака и выборку всех остальных признаков-----------------------------------------


train_scores, test_scores, lam = [], [], 1000000 #тренировочный и тестовый массивы для записи ошибок
MR = MultipleRegressionReg() #инициализация класса множественной регрессии

#L1---------------------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 0, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = MR.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="L1", lam=l)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    trainMSE = MR.MSE(x_i, y_train) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки
    train_scores.append(trainMSE)

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    testMSE = MR.MSE(x_i, y_test) #добавление значения средней квадратичной ошибки в массив для тестовой выборки
    test_scores.append(testMSE)

    trainTestDiffernce = abs(trainMSE - testMSE)
    if trainTestDiffernce < lam:
        lam = trainTestDiffernce
print(f"Лямбда для L1 - {lam}")

#plt.plot(train_scores, 'g')
#plt.plot(test_scores, 'r')
#plt.xlabel('L1')
#L1--------------------------------------------------------------------------------------------------------------------------


#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------
x_i = x_train.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тренировочной выборки(L1) - {MR.MSE(x_i, y_train)}")

x_i = x_test.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тестовой выборки(L1) - {MR.MSE(x_i, y_test)}")
#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------


train_scores, test_scores, lam = [], [], 1000000 #тренировочный и тестовый массивы для записи ошибок
MR = MultipleRegressionReg() #инициализация класса множественной регрессии


#L2---------------------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 0, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = MR.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="L2", lam=l)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    trainMSE = MR.MSE(x_i, y_train) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки
    train_scores.append(trainMSE)

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    testMSE = MR.MSE(x_i, y_test) #добавление значения средней квадратичной ошибки в массив для тестовой выборки
    test_scores.append(testMSE)

    trainTestDiffernce = abs(trainMSE - testMSE)
    if trainTestDiffernce < lam:
        lam = trainTestDiffernce
print(f"Лямбда для L2 - {lam}")


#plt.plot(train_scores, 'g')
#plt.plot(test_scores, 'r')
#plt.xlabel('L2')
#L2--------------------------------------------------------------------------------------------------------------------------


#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------
x_i = x_train.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тренировочной выборки(L2) - {MR.MSE(x_i, y_train)}")

x_i = x_test.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тестовой выборки(L2) - {MR.MSE(x_i, y_test)}")
#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------


train_scores, test_scores, lam = [], [], 1000000 #тренировочный и тестовый массивы для записи ошибок
MR = MultipleRegressionReg() #инициализация класса множественной регрессии


#ElasticNet---------------------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 0, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = MR.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="ElasticNet", lam=l, rho=0.5)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    trainMSE = MR.MSE(x_i, y_train) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки
    train_scores.append(trainMSE)

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    testMSE = MR.MSE(x_i, y_test) #добавление значения средней квадратичной ошибки в массив для тестовой выборки
    test_scores.append(testMSE)

    trainTestDiffernce = abs(trainMSE - testMSE)
    if trainTestDiffernce < lam:
        lam = trainTestDiffernce
print(f"Лямбда для EN(0.5) - {lam}")

#plt.plot(train_scores, 'g')
#plt.plot(test_scores, 'r')
#plt.xlabel('ElasticNet')
#ElasticNet--------------------------------------------------------------------------------------------------------------------------

#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------
x_i = x_train.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тренировочной выборки(EN) - {MR.MSE(x_i, y_train)}")

x_i = x_test.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тестовой выборки(EN) - {MR.MSE(x_i, y_test)}")
#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------


train_scores, test_scores, lam = [], [], 1000000 #тренировочный и тестовый массивы для записи ошибок
MR = MultipleRegressionReg() #инициализация класса множественной регрессии



#ElasticNet---------------------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 0, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = MR.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="ElasticNet", lam=l, rho=0.1)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    trainMSE = MR.MSE(x_i, y_train) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки
    train_scores.append(trainMSE)

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    testMSE = MR.MSE(x_i, y_test) #добавление значения средней квадратичной ошибки в массив для тестовой выборки
    test_scores.append(testMSE)

    trainTestDiffernce = abs(trainMSE - testMSE)
    if trainTestDiffernce < lam:
        lam = trainTestDiffernce
print(f"Лямбда для EN(0.1) - {lam}")

#plt.plot(train_scores, 'g')
#plt.plot(test_scores, 'r')
#plt.xlabel('ElasticNet')
#ElasticNet--------------------------------------------------------------------------------------------------------------------------

#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------
x_i = x_train.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тренировочной выборки(EN) - {MR.MSE(x_i, y_train)}")

x_i = x_test.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тестовой выборки(EN) - {MR.MSE(x_i, y_test)}")
#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------


train_scores, test_scores, lam = [], [], 1000000 #тренировочный и тестовый массивы для записи ошибок
MR = MultipleRegressionReg() #инициализация класса множественной регрессии


#ElasticNet---------------------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 0, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = MR.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="ElasticNet", lam=l, rho=0.9)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    trainMSE = MR.MSE(x_i, y_train) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки
    train_scores.append(trainMSE)

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    testMSE = MR.MSE(x_i, y_test) #добавление значения средней квадратичной ошибки в массив для тестовой выборки
    test_scores.append(testMSE)

    trainTestDiffernce = abs(trainMSE - testMSE)
    if trainTestDiffernce < lam:
        lam = trainTestDiffernce
print(f"Лямбда для EN(0.9) - {lam}")

#plt.plot(train_scores, 'g')
#plt.plot(test_scores, 'r')
#plt.xlabel('ElasticNet')
#ElasticNet-------------------------------------------------------------------------------------------------------------------------


#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------
x_i = x_train.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тренировочной выборки(EN) - {MR.MSE(x_i, y_train)}")

x_i = x_test.copy()
x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
print(f"Средняя квадратичная ошибка тестовой выборки(EN) - {MR.MSE(x_i, y_test)}")
#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------



