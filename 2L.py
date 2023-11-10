import time
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from LR import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data/insclass_train.csv")

# Выборка нужных столбцов для признаков и результирующего признака
features = data.drop(['target', "variable_7", "variable_9", "variable_15"], axis=1)
target = data.drop(['variable_1', 'variable_2', 'variable_3',
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


# Разделение признаков и результирующего признака на тестовые и тренировочные выборки
features_train, features_test, target_train, target_test = train_test_split(
     features, target,
     stratify=target,
     test_size=0.25,
     random_state=42,
)
# Разделение признаков и результирующего признака на тестовые и тренировочные выборки

# Преобразование категориальных признаков в числовые
TE = TargetEncoder(cols=['variable_1', 'variable_5', 'variable_20', 'variable_21', 'variable_22', 'variable_28']).fit(features_train, target_train)
TE.transform(features_train)
features_train = TE.transform(features_train)
features_test = TE.transform(features_test)
# Преобразование категориальных признаков в числовые

# Выбор только числовых признаков
onlyNumColumns = features_train.select_dtypes(include='number').columns
# Выбор только числовых признаков

# Замена всех NaN признаков на среднее значение
SI = SimpleImputer(missing_values=np.nan, strategy='mean')
SI.fit(features_train[onlyNumColumns])
features_train[onlyNumColumns] = SI.transform(features_train[onlyNumColumns])
features_test[onlyNumColumns] = SI.transform(features_test[onlyNumColumns])
# Замена всех NaN признаков на среднее значение

# Нормализация масштаба данных (нейтрализация переполнения при использовании экспоненциальной функии)
MMS = MinMaxScaler()
MMS.fit(features_train[onlyNumColumns])
features_train[onlyNumColumns] = MMS.transform(features_train[onlyNumColumns])
features_test[onlyNumColumns] = MMS.transform(features_test[onlyNumColumns])
# Нормализация масштаба данных (нейтрализация переполнения при использовании экспоненциальной функии)

# Добавление столбца единиц
features_train['var_0'] = np.ones((features_train.shape[0]))
features_test['var_0'] = np.ones((features_test.shape[0]))
# Добавление столбца единиц

# Обновленный выбор только номерных столбцов
onlyNumColumns = features_train.select_dtypes(include='number').columns
# Обновленный выбор только номерных столбцов

# Разворот одномерного массива в двумерный
target_train = np.array(target_train).reshape(-1, 1)
target_test = np.array(target_test).reshape(-1, 1)
# Разворот одномерного массива в двумерный

LR = LogisticRegressionGD()

start = time.time()

stepsGR, errorsGR = LR.fit(
    features_train[onlyNumColumns],
    target_train,
    alpha=0.01,
    max_steps=5000,
    Rtype="GR"
)
print(f"Точность модели (GR): {LR.accuracy(features_test[onlyNumColumns], target_test)[0]}, время расчета - {(time.time() - start) // 60} мин {(time.time() - start) % 60} сек")


stepsGD, errorsGD = LR.fit(
    features_train[onlyNumColumns],
    target_train,
    alpha=0.01,
    max_steps=5000,
    Rtype="GD"
)
print(f"Точность модели (GD): {LR.accuracy(features_test[onlyNumColumns], target_test)[0]}, время расчета - {(time.time() - start) // 60} мин {(time.time() - start) % 60} сек")

# Использования ресемплера
rus = RandomUnderSampler()
features_train_resampled, target_train_resampled = rus.fit_resample(features_train, target_train)
# Использования ресемплера

target_train_resampled = np.array(target_train_resampled).reshape(-1, 1)

stepsGRS, errorsGRS = LR.fit(
    features_train_resampled[onlyNumColumns],
    target_train_resampled,
    alpha=0.01,
    max_steps=5000,
    Rtype="GR"
)
print(f"Точность модели (GRS): {LR.accuracy(features_test[onlyNumColumns], target_test)[0]}, время расчета - {(time.time() - start) // 60} мин {(time.time() - start) % 60} сек")

stepsGDS, errorsGDS = LR.fit(
    features_train_resampled[onlyNumColumns],
    target_train_resampled,
    alpha=0.01,
    max_steps=5000,
    Rtype="GD"
)
print(f"Точность модели (GDS): {LR.accuracy(features_test[onlyNumColumns], target_test)[0]}, время расчета - {(time.time() - start) // 60} мин {(time.time() - start) % 60} сек")

fig = plt.figure()
ax = fig.add_subplot()

ax.set_title('Изменение значений функций потерь', fontsize=15)

ax.set_xlabel('Шаг')
ax.set_ylabel('Значение функции ошибки')
graph = ax.plot(stepsGRS, errorsGRS, label='GRS')
graph = ax.plot(stepsGDS, errorsGDS, label='GDS')
graph = ax.plot(stepsGR, errorsGR, label='GR')
graph = ax.plot(stepsGD, errorsGD, label='GD')

handles, labels = ax.get_legend_handles_labels()
graphS = ax.legend(handles, labels)
plt.show()


