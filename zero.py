train_scores, test_scores = [], [] #тренировочный и тестовый массивы для записи ошибок

#L2--------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 10, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = reg_regr.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="L2", lam=l)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    train_scores.append(reg_regr.MSE(x_i, y_train)) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    test_scores.append(reg_regr.MSE(x_i, y_test)) #добавление значения средней квадратичной ошибки в массив для тестовой выборки

print(f"Средние квадратичные ошибки тренировочной выборки (L2) - {train_scores}")
print(f"Средние квадратичные ошибки тестовой выборки (L2) - {test_scores}")
plt.plot(train_scores, 'g') #построение графика по массиву средних квадратичных ошибок для тренировочной выборки
plt.plot(test_scores, 'r') #построение графика по массиву средних квадратичных ошибок для тестовой выборки
plt.show()
#L2--------------------------------------------------------------------------------------------------------------------

train_scores, test_scores = [], [] #тренировочный и тестовый массивы для записи ошибок

#L2--------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 10, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = reg_regr.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="L2", lam=l)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    train_scores.append(reg_regr.MSE(x_i, y_train)) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    test_scores.append(reg_regr.MSE(x_i, y_test)) #добавление значения средней квадратичной ошибки в массив для тестовой выборки

print(f"Средние квадратичные ошибки тренировочной выборки (L2 вторая волна) - {train_scores}")
print(f"Средние квадратичные ошибки тестовой выборки (L2 вторая волна) - {test_scores}")
plt.plot(train_scores, 'g') #построение графика по массиву средних квадратичных ошибок для тренировочной выборки
plt.plot(test_scores, 'r') #построение графика по массиву средних квадратичных ошибок для тестовой выборки
plt.show()
#L2--------------------------------------------------------------------------------------------------------------------

#ElasticNet 1--------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 10, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = reg_regr.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="ElasticNet", lam=l, rho=1)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    train_scores.append(reg_regr.MSE(x_i, y_train)) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    test_scores.append(reg_regr.MSE(x_i, y_test)) #добавление значения средней квадратичной ошибки в массив для тестовой выборки

print(f"Средние квадратичные ошибки тренировочной выборки (EL 1) - {train_scores}")
print(f"Средние квадратичные ошибки тестовой выборки (EL 1) - {test_scores}")
plt.plot(train_scores, 'g') #построение графика по массиву средних квадратичных ошибок для тренировочной выборки
plt.plot(test_scores, 'r') #построение графика по массиву средних квадратичных ошибок для тестовой выборки
plt.show()
#ElasticNet 1--------------------------------------------------------------------------------------------------------------------

train_scores, test_scores = [], [] #тренировочный и тестовый массивы для записи ошибок

#ElasticNet 0.5--------------------------------------------------------------------------------------------------------------------
for l in np.logspace(-5, 10, 15): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    steps, errors = reg_regr.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="ElasticNet", lam=l, rho=0.5)

    x_i = x_train.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    train_scores.append(reg_regr.MSE(x_i, y_train)) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки

    x_i = x_test.copy()
    x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    test_scores.append(reg_regr.MSE(x_i, y_test)) #добавление значения средней квадратичной ошибки в массив для тестовой выборки

print(f"Средние квадратичные ошибки тренировочной выборки (EL 0.5) - {train_scores}")
print(f"Средние квадратичные ошибки тестовой выборки (EL 0.5) - {test_scores}")
plt.plot(train_scores, 'g') #построение графика по массиву средних квадратичных ошибок для тренировочной выборки
plt.plot(test_scores, 'r') #построение графика по массиву средних квадратичных ошибок для тестовой выборки
plt.show()
#ElasticNet 0.5--------------------------------------------------------------------------------------------------------------------

train_scores, test_scores = [], [] #тренировочный и тестовый массивы для записи ошибок