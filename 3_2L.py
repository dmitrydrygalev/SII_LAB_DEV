import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from utils import *
from LR import *

data = pd.read_csv("./data/housing.csv")

array = []
for i in range(103):
    array.append(i)

Y = data.drop(data.columns[:-1], axis=1)  # выборка только результирующего признака
x = data.drop('Y', axis=1)  # выборка всех признаков кроме результирующего
print(Y)
print(x)

for column in x:
    amount = 0
    for value in x[column]:
        if value == 0:
            amount = amount + 1
    if amount > 300:
        x = x.drop(str(column), axis=1)

x_train, Y_train, x_test, Y_test = trainTestSpliter(x, Y, 0.25, 123)
print(x_train, Y_train, x_test, Y_test)

Y_train[0] = Y_train[0].astype(int)
Y_test[0] = Y_test[0].astype(int)
Y_train = Y_train.rename(columns={0: "Y"})
Y_test = Y_test.rename(columns={0: "Y"})
print(Y_train)
print(Y_test)

rus = RandomUnderSampler()
x_train_resampled, Y_train_resampled = rus.fit_resample(x_train, Y_train)
print(x_train_resampled, Y_train_resampled)

# Добавление столбца единиц--------------------------------------------------
#x_train_resampled['var_0'] = np.ones((x_train_resampled.shape[0]))
#x_test['var_0'] = np.ones((x_test.shape[0]))
# Добавление столбца единиц--------------------------------------------------


Y_train_resampled = Y_train_resampled.values.reshape(-1, 1)
Y_test = Y_test.values.reshape(-1, 1)
print(Y_train)
print(Y_test)

LR = LogisticRegressionGD()

stepsGRS, errorsGRS = LR.fit(
    x_train_resampled,
    Y_train_resampled,
    alpha=0.01,
    max_steps=5000,
    Rtype="GR"
)
print(f"Стандартное значение accuracy: {LR.accuracy(x_test, Y_test)[0]}")
print(f"Стандартное значение F1: {LR.f1_score(x_test, Y_test)}")
