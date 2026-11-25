import numpy as np
class SVM: 
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None
    def compute_margin(self, X, y, w, b):
        return y * (np.dot(X, w) + b)
    def loss_function(self, X, y, w, b, margin):
        hinge_loss = np.maximum(0, 1 - self.compute_margin(X, y, w, b))
        regularization_loss = 0.5 * np.dot(w, w)
        if margin == "hard":
            if np.all(hinge_loss == 0):
                return regularization_loss
            else:
                return np.inf
        else:
            return regularization_loss + self.C * np.sum(hinge_loss)
    def SGD(self, X, y, learning_rate=0.01, epochs=1000, margin = "soft"):
        m, n = X.shape
        w = np.zeros(X.shape[1])
        b = 0
        for _ in range(epochs):
            for i in range(X.shape[0]):
                if y[i] * (np.dot(w, X[i]) + b) < 1:
                    w += learning_rate * (self.C * m * y[i] * X[i] - w)
                    b += learning_rate * self.C * m *y[i]
                else: 
                    w -= learning_rate * w
        self.w = w
        self.b = b
    def compute_error(self, y, alphas, b, K, i):
        return np.dot((alphas * y), K[:, i]) + b - y[i]
    def SMO(self, X, y, tol=1e-3, max_passes=5, margin = "hard", max_iteration = 1000):
        m, n = X.shape
        if margin == "soft":
            C = self.C
        elif margin == "hard":
            C = np.inf
        else:
            raise ValueError("Invalid margin type. Please choose 'soft' or 'hard'.")
        b = 0.0
        K = X @ X.T
        passes = 0
        alphas = np.zeros(m)
        iteration = 0
        while passes < max_passes and iteration < max_iteration:
            num_changed_alphas = 0
            for i in range(m): 
                Ei = self.compute_error(y, alphas, b, K, i)
                if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                    j = np.random.choice(m)
                    while j == i:
                        j = np.random.choice(m)
                    Ej = self.compute_error(y, alphas, b, K, j)
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - C)
                        H = min(C, alphas[i] + alphas[j])
                    if H == L:
                        continue
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue
                    alphas[j] += y[j] * (Ei - Ej) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        alphas[j] = alpha_j_old
                        continue
                    num_changed_alphas += 1
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j]) 
                    delta_i = alphas[i] - alpha_i_old
                    delta_j = alphas[j] - alpha_j_old
                    b1 = b - Ei - y[i] * delta_i * K[i, i] - y[j] * delta_j * K[i, j]
                    b2 = b - Ej - y[i] * delta_i * K[i, j] - y[j] * delta_j * K[j, j]
                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
            if num_changed_alphas == 0:
                passes += 1 
            else: 
                passes = 0
            iteration += 1
        self.w = np.dot(alphas * y, X)
        self.b = b
        self.alphas = alphas
                        
    def fit(self, X, y, optimizer="smo", margin="soft", **kwargs):

        if optimizer == "smo":
            self.SMO(X, y, margin=margin, **kwargs)

        elif optimizer == "sgd":
            if margin != "soft":
                raise ValueError("SGD only supports soft margin")

            self.SGD(X, y, margin=margin, **kwargs)
        else:
            raise ValueError("Invalid optimizer")
        return self
    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
    def score(self, X, y):
        return np.mean(self.predict(X) == y)