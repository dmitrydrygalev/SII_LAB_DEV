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

#train_scores, test_scores = [], [] #тренировочный и тестовый массивы для записи ошибок
#MR = MultipleRegression() #инициализация класса множественной регрессии

#L1---------------------------------------------------------------------------------------------------------------------------------
#for l in np.logspace(-2, 1, 10): #итерация l среди логарифмической сетки размером 15 точек от -5 до 10

    #steps, errors = MR.fit(x_train, y_train, intercept=True, alpha=0.01, epsylon=0.00001, max_steps=5000, reg="L1", lam=l)

    #train_predictions = MR.predict(x_train)

    #test_predictions = MR.predict(x_test)

    #train_scores.append(np.mean((y_train - train_predictions)))

    #test_scores.append(np.mean((y_test - test_predictions)))

#print(f"Тренировочная выборка по L1(mod) -{train_scores}")
#print(f"Тестовая выборка по L1(mod) -{test_scores}")
    #x_i = x_train.copy()
    #x_i.insert(0, "intercept", np.ones((x_i.shape[0], 1)), allow_duplicates=True)
    #train_scores.append(MR.MSE(x_i, y_train)) #добавление значения средней квадратичной ошибки в массив для тренировочной выборки

    #x_i = x_test.copy()
    #x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
    #test_scores.append(MR.MSE(x_i, y_test)) #добавление значения средней квадратичной ошибки в массив для тестовой выборки

#plt.plot(train_scores) #построение графика по массиву средних квадратичных ошибок для тренировочной выборки
#plt.plot(test_scores) #построение графика по массиву средних квадратичных ошибок для тестовой выборки
#plt.show()
#L1--------------------------------------------------------------------------------------------------------------------------

# TODO: РЕАЛИЗОВАТЬ МЕТРИКИ L2 B ELASTICNET, НЕ СМОТРЯ НА ТО ЧТО L1 ОКАЗАЛСЯ ЭФФЕКТИВНЕЕ

#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------
#x_i = x_train.copy()
#x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
#print(f"Средняя квадратичная ошибка тренировочной выборки - {MR.MSE(x_i, y_train)}")

#x_i = x_test.copy()
#x_i.insert(0, "intercept", np.ones((x_i.shape[0],1)), allow_duplicates=True)
#print(f"Средняя квадратичная ошибка тестовой выборки - {MR.MSE(x_i, y_test)}")
#Проверка расхождения средних квадрачтиных ошибок между тестовой и тренировочной выборками-----------------------------------

#def find_best_lambda(x_train, y_train, x_test, y_test, reg_type, rho=0.1):
    #lambdas = np.logspace(-5, 10, 15)
    #train_mse = []
    #test_mse = []
    #for l in lambdas:
        #if reg_type == "L1":
            #model = Lasso(alpha=l)
        #elif reg_type == "L2":
           # model = Ridge(alpha=l)
        #elif reg_type == "ElasticNet":
            #model = ElasticNet(alpha=l, l1_ratio=rho)

        #model.fit(x_train, y_train)

        #train_predictions = model.predict(x_train)

        #test_predictions = model.predict(x_test)

        #train_mse.append(np.mean((y_train - train_predictions)))

        #test_mse.append(np.mean((y_test - test_predictions)))

data1 = pd.read_csv("data/insclass_train.csv")

# Выборка нужных столбцов для признаков и результирующего признака
features = data1.drop(['target', "variable_7", "variable_9", "variable_15"], axis=1)
target = data1.drop(['variable_1', 'variable_2', 'variable_3',
                    'variable_4', 'variable_5', 'variable_6',
                    'variable_7', 'variable_8', 'variable_9',
                    'variable_10', 'variable_11', 'variable_12',
                    'variable_13', 'variable_14', 'variable_15',
                    'variable_16', 'variable_17', 'variable_18',
                    'variable_19', 'variable_20', 'variable_21',
                    'variable_22', 'variable_23', 'variable_24',
                    'variable_25', 'variable_26', 'variable_27',
                    'variable_28'], axis=1)

for index, row in target.iterrows():
    if pd.isna(row['target']):
        target.iat[index, 0] = 0
# Выборка нужных столбцов для признаков и результирующего признака

print(target)