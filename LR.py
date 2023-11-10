import numpy as np


class LogisticRegressionGD:

    def __init__(self, threshold=0.5):
        self.a = np.zeros(2).reshape(1, 2)
        self._estimator_type = "classifier"
        self.threshold = threshold

    def sigmoid(self, x):
        """ Функция активации - сигмоид """
        return 1/(1 + np.exp(-x @ self.a))

    def predict_proba(self, x):
        """ Вероятность принадлежности к классу 1 """
        return self.sigmoid(x)

    def predict(self, x):
        """ Прогноз класса """
        return (self.predict_proba(x) > self.threshold).astype(int)

    def coefs(self):
        """ Значения весов """
        return self.a

    def LogLikelihood(self, x, Y):
        """ Функция потерь - логарифмическая функция правдоподобия """
        return (Y * np.log(self.predict_proba(x) + 1e-7) + (1 - Y) * np.log(1 + 1e-7 - self.predict_proba(x))).mean()

    def CrossEntropy(self, x, Y):
        """ Функция потерь - перекрестная энтропия """
        return (-Y * np.log(self.predict_proba(x)) - (1 - Y) * np.log(1 - self.predict_proba(x))).mean()

    def accuracy(self, x, Y):
        """ Точность """
        return (self.predict(x) == Y).mean()

    def f1_score(self, x, Y):
        predictions = self.predict(x)
        true_positives = np.sum(np.logical_and(predictions > 0.5, Y == 1))
        false_positives = np.sum(np.logical_and(predictions > 0.5, Y == 0))
        false_negatives = np.sum(np.logical_and(predictions <= 0.5, Y == 1))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return 2 * (precision * recall) / (precision + recall)

    def fit(self, x, Y, alpha = 0.01, epsylon = 0.01, max_steps = 10000, Rtype = "GR"):
        """ Обучение модели с помощью градиентного спуска """
        self.a = np.zeros (x.shape[1]).reshape(x.shape[1],1)
        steps, errors = [], []
        step = 0
        for _ in range(max_steps):
            if Rtype == "GR":
                new_error = self.LogLikelihood(x, Y)
                dT_a = x.T @(Y - self.predict_proba(x)) / x.shape[0]
                self.a += alpha*dT_a
            elif Rtype == "GD":
                new_error = self.CrossEntropy(x, Y)
                dT_a = -x.T @(Y - self.predict_proba(x)) / x.shape[0]
                self.a -= alpha*dT_a
            step += 1
            steps.append(step)
            errors.append(new_error)
        return steps, errors
