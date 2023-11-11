import numpy as np
class LogisticRegressionGD:
    def __init__(self, threshold=0.5):
        self.a = np.zeros(2).reshape(1, 2)
        self._estimator_type = "classifier"
        self.threshold = threshold
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x @ self.a))
    def predict_proba(self, x):
        return self.sigmoid(x)
    def predict(self, x):
        return (self.predict_proba(x) > self.threshold).astype(int)
    def coefs(self):
        return self.a
    def LogLikelihood(self, x, Y):
        return (Y * np.log(self.predict_proba(x) + 1e-7) + (1 - Y) * np.log(1 + 1e-7 - self.predict_proba(x))).sum()
    def CrossEntropy(self, x, Y):
        return (-Y * np.log(self.predict_proba(x)) - (1 - Y) * np.log(1 - self.predict_proba(x))).sum()
    def accuracy(self, x, Y):
        return (self.predict(x) == Y).mean()
    def fit(self, x, Y, alpha = 0.01, epsylon = 0.01, max_steps = 10000, Rtype = "LL"):
        self.a = np.zeros(x.shape[1]).reshape(x.shape[1],1)
        steps, errors = [], []
        step = 0
        for _ in range(max_steps):
            if Rtype == "LL":
                new_error = self.LogLikelihood(x, Y)
                dT_a = x.T @(Y - self.predict_proba(x)) / x.shape[0]
                self.a += alpha*dT_a
            elif Rtype == "CE":
                new_error = self.CrossEntropy(x, Y)
                #display(new_error)
                dT_a = -x.T @(Y - self.predict_proba(x)) / x.shape[0]
                self.a -= alpha*dT_a
            step += 1
            steps.append(step)
            errors.append(new_error)
            #if new_error < epsylon:
            #    break
        return steps, errors
