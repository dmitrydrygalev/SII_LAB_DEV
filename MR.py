import numpy as np


class MultipleRegressionReg(object):
    def __init__(self):
        self.a = np.zeros(1)
        self.intercept = True

    def predict(self, x):
        return x @ self.a

    def MSE(self, x, Y):
        return \
        (((Y.to_numpy() - self.predict(x).to_numpy()).T @ (Y.to_numpy() - self.predict(x).to_numpy())) / x.shape[0])[0][
            0]

    def MAE(self, x, Y):
        return abs((Y.to_numpy() - self.predict(x).to_numpy())).mean()

    def MAPE(self, x, Y):
        return abs((Y.to_numpy() - self.predict(x).to_numpy()) / Y).mean().to_real()

    def coefs(self):
        return self.a

    def fit(self, x, Y, alpha=0.001, epsylon=0.01, max_steps=5000, intercept=True, reg="No", lam=0, rho=0):
        self.intercept = intercept
        self.x = x.copy()
        self.Y = Y.copy()
        if intercept:
            self.x.insert(0, "intercept", np.ones((self.x.shape[0], 1)), allow_duplicates=True)
        m = len(self.x)
        self.a = np.zeros(self.x.shape[1]).reshape(self.x.shape[1], 1)
        steps, errors = [], []
        step = 0
        for _ in range(max_steps):
            dT_a = -2 * self.x.T @ (self.Y.to_numpy() - self.predict(self.x).to_numpy()) / m
            if reg == "L1":
                dT_a += lam * (np.sign(self.a))
            elif reg == "L2":
                dT_a += 2 * lam * ((self.a))
            elif reg == "ElasticNet":
                dT_a += rho * lam * (np.sign(self.a)) + 2 * (1 - rho) * lam * ((self.a))

            self.a -= alpha * dT_a
            new_error = self.MSE(self.x, self.Y)
            step += 1
            steps.append(step)
            errors.append(new_error)
            if new_error < epsylon:
                break
        return steps, errors
