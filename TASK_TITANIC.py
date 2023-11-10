import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sbn
import matplotlib.pyplot as plt

data = pd.read_csv('data/titanic.csv')  # чтение данных из файла
data.dropna(inplace=True)  # удаление строк с отсутствующими значениями

oe = OneHotEncoder(sparse=False)

sex = pd.get_dummies(data['Sex'], drop_first=True) # преобразование категориального признака "пол" в числовое значение
embark = pd.get_dummies(data['Embarked'], drop_first=True) # преобразование категориального признака "порт отправления" в числовое значение

sex['male'] = sex['male'].replace({True: 1, False: 0}, regex=True)
embark[['Q', 'S']] = embark[['Q', 'S']].replace({True: 1, False: 0}, regex=True)

data.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True) # удаление всех не нужных столбцов

data = pd.concat([data, sex, embark], axis=1) # добавелние новых столбцов в dataframe

si = SimpleImputer(strategy='mean')

data = pd.DataFrame(si.fit_transform(data), columns=data.columns)

# 1 --------------------------------------------------------------------------------------------
features = data.drop('Survived', axis=1)
target = data['Survived']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=42)

LR = LogisticRegression()
LR.fit(features_train, target_train)
target_pred = LR.predict(features_test)

accuracy = accuracy_score(target_test, target_pred)

print(f'Accuracy: {accuracy}')
# 1 --------------------------------------------------------------------------------------------


# 3 --------------------------------------------------------------------------------------------
# Какова доля выживших после крушения пассажиров?
data = pd.read_csv('data/titanic.csv')

sbn.countplot(x='Survived', data=data, palette=['green', 'orange'])
plt.show()

sbn.countplot(x='Survived', data=data, hue='Sex', palette=['green', 'orange'])
plt.show()
# 3 --------------------------------------------------------------------------------------------


# 4 --------------------------------------------------------------------------------------------
# Сколько пассажиров ехало в каждом классе? Кого было больше в самом многолюдном классе — мужчин или женщин?
sbn.countplot(x='Pclass', data=data, palette=['green', 'orange'])
sbn.countplot(x='Pclass', data=data, hue='Sex', palette=['green', 'orange'])
# 4 --------------------------------------------------------------------------------------------

# 5 --------------------------------------------------------------------------------------------
# Все ли признаки несут в себе полезную информацию? Почему? Избавьтесь от ненужных столбцов.
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# 5 --------------------------------------------------------------------------------------------


# 6 --------------------------------------------------------------------------------------------
# Посчитайте, насколько сильно коррелируют друг с другом цена за билет и возраст пассажиров.
# Также проверьте наличие этой зависимости визуально
sbn.scatterplot(x='Fare', y='Age', data=data)
plt.show()
corr = data['Fare'].corr(data['Age'])
print(f'Коэффициент корреляции - {corr}')
# 6 --------------------------------------------------------------------------------------------


# 7 --------------------------------------------------------------------------------------------
#Правда ли, что чаще выживали пассажиры с более дорогими билетами? А есть ли зависимость выживаемости от класса?
sbn.boxplot(x='Survived', y='Fare', data=data, palette=['green', 'orange'])
sbn.boxplot(x='Survived', y='Pclass', data=data, palette=['green', 'orange'])
plt.show()
# 7 --------------------------------------------------------------------------------------------


# 8 --------------------------------------------------------------------------------------------
# Какова связь между стоимостью билета и портом отправления?
# Выведите минимальную, среднюю и максимальную сумму, которую заплатили пассажиры за проезд
sbn.boxplot(x='Embarked', y='Fare', data=data, palette=['green', 'orange', 'purple'])
plt.show()
min_fare = data.groupby('Embarked')['Fare'].min()
mean_fare = data.groupby('Embarked')['Fare'].mean()
max_fare = data.groupby('Embarked')['Fare'].max()
print(f'{min_fare}\n{max_fare}\n{mean_fare}')
# 8 --------------------------------------------------------------------------------------------


# 9 --------------------------------------------------------------------------------------------
# Выведите гистограммы, показывающие распределения стоимостей билетов в зависимости от места посадки
sbn.histplot(x='Fare', hue='Embarked', data=data, palette=['green', 'orange', 'purple'])
plt.show()
# 9 --------------------------------------------------------------------------------------------

# 10 --------------------------------------------------------------------------------------------
# Оцените репрезентативность представленной выборки. Сколько всего было пассажиров Титаника?
# Сколько из них выжило?
# Какую долю составляет представленный набор данных от всей генеральной совокупности?
passengers_amount = len(data)
passengers_survived = data['Survived'].sum()
isRepresent = len(data) / passengers_amount
# 10 --------------------------------------------------------------------------------------------


# 11 --------------------------------------------------------------------------------------------
# Разделите выборку на тестовую и обучающую части при помощи train_test_split().
# Изобразите на графиках распределение некоторых атрибутов и целевой переменной.
# Насколько однородно получившееся разбиение?
features = data.drop('Survived', axis=1)
target = data['Survived']
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=42)
# 11 --------------------------------------------------------------------------------------------


# 13 --------------------------------------------------------------------------------------------
# Удалите лишние объекты мажоритарного класса (выбранные случайно)
# Определите мажоритарный класс
# Индексы объектов мажоритарного класса
# Выберите случайные индексы для удаления
# Удалите объекты мажоритарного класса
majorСlass = data['Survived'].mode().iloc[0]
majorIndexs = data[data['Survived'] == majorСlass].index
majorIndexsForRemove = np.random.choice(majorIndexs, size=int(0.5 * len(majorIndexs)), replace=False)
majorClassObjectsRemove = data.drop(majorIndexsForRemove)
# 13 --------------------------------------------------------------------------------------------


# 14 --------------------------------------------------------------------------------------------
# Добавьте в выборку дубликаты миноритарного класса
# Определите миноритарный класс
# Индексы объектов миноритарного класса
# Добавьте дубликаты миноритарного класса
minorClass = 1 - majorСlass
minorIndexs = data[data['Survived'] == minorClass].index
cloneMinorClass = pd.concat([data, data.loc[minorIndexs]])
# 14 --------------------------------------------------------------------------------------------


# 15 --------------------------------------------------------------------------------------------
# Проведите исследование эффективности простой модели классификации до и после данных преобразований.
modelDefault = LogisticRegression()
modelDefault.fit(features_train, target_train)
targetPredDefault = modelDefault.predict(features_test)
accuracyDefault = accuracy_score(features_test, targetPredDefault)

modelModify = LogisticRegression()
modelModify.fit(majorClassObjectsRemove.drop('Survived', axis=1), majorClassObjectsRemove['Survived'])
targetPredModify = modelModify.predict(features_test)
accuracyModify = accuracy_score(features_test, targetPredModify)
# 15 --------------------------------------------------------------------------------------------


# 16 --------------------------------------------------------------------------------------------
# Постройте корреляционную матрицу признаков после преобразования данных.
# Сделайте вывод о наличии либо отсутствии мультиколлинеарности признаков.
# Построение корреляционной матрицы
# Отфильтровать только числовые данные
# Построить тепловую карту корреляций
# Построить корреляционную матрицу
numeric_data = majorClassObjectsRemove.select_dtypes(include=np.number)

correlation_matrix = numeric_data.corr()
plt.figure(figsize=(20, 6))
sbn.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=.5)
plt.title("Корреляционная матрица после трансформации")
plt.show()
# 16 --------------------------------------------------------------------------------------------


# 17 --------------------------------------------------------------------------------------------
# Проведите группировку данных по значению возраста.
# Введите новый признак "возрастная категория", значениями которой будут "ребенок", "взрослый", "старик".
# Проведите анализ эффективности данного признака.
# Введите новый признак "возрастная категория"
bins = [0, 18, 60, np.inf]
labels = ['Ребенок', 'Взрослый', 'Пожилой']

majorClassObjectsRemove['Age_Category'] = pd.cut(majorClassObjectsRemove['Age'], bins=bins, labels=labels)

sbn.countplot(x='Age_Category', hue='Survived', data=majorClassObjectsRemove, palette=['green', 'orange'])
plt.title("Выжившие по возрастной категории")
plt.show()
# 17 --------------------------------------------------------------------------------------------














