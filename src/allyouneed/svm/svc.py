import numpy as np
import cvxopt
from ..base import BaseClassifier


class SVC(BaseClassifier):
    def __init__(self, kernel='linear', gamma=None, degree=3, coef0=0.0, C=1.0,
                 optimizer='cvxopt', max_iter=200, tol=0.001, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.support_vectors_ = None
        self.support_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.coef_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.isfitted = False

    def _get_kernel_fn(self):
        if self.kernel == 'linear':
            return lambda x_i, x_j: np.dot(x_i, x_j)
        elif self.kernel == 'rbf':
            return lambda x_i, x_j: np.exp(-self.gamma * np.dot(x_i - x_j, x_i - x_j))
        elif self.kernel == 'poly':
            return lambda x_i, x_j: (self.gamma * np.dot(x_i, x_j) + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return lambda x_i, x_j: np.tanh(self.gamma * np.dot(x_i, x_j) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_kernel_matrix(self, X, kernel_fn):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = kernel_fn(X[i], X[j])

        return K

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVC only supports binary classification. Use MulticlassSVC for multiclass.")

        # Convert labels to {-1, +1}
        y_binary = np.where(y == self.classes_[0], -1, 1)

        # Set gamma for RBF, polynomial, and sigmoid kernels
        if self.gamma is None:
            self.gamma = 1.0 / (n_features * X.var())

        # Appropriate optimizer
        if self.optimizer == 'cvxopt':
            self._fit_cvxopt(X, y_binary, n_samples, n_features)
        elif self.optimizer == 'smo':
            self._fit_smo(X, y_binary, n_samples, n_features)
        elif self.optimizer == 'pegasos':
            self._fit_pegasos(X, y_binary, n_samples, n_features)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        self.isfitted = True
        return self

    def _fit_cvxopt(self, X, y_binary, n_samples, n_features):
        kernel_fn = self._get_kernel_fn()

        # Compute kernel matrix K[i, j] = K(x_i, x_j)
        K = self._compute_kernel_matrix(X, kernel_fn)

        # max{L_D(Lambda)} can be rewritten as
        #   min{1/2 Lambda^T H Lambda - 1^T Lambda}
        #       s.t. -lambda_i <= 0
        #       s.t. lambda_i <= C
        #       s.t. y^T Lambda = 0
        # where H[i, j] = y_i y_j K(x_i, x_j)

        # Standard form to CVXOPT's quadratic programming solver:
        #   min{1/2 x^T P x + q^T x}
        #       s.t. G x <= h
        #       s.t. A x = b
        P = cvxopt.matrix(np.outer(y_binary, y_binary) * K)
        q = cvxopt.matrix(-np.ones(n_samples))

        # Constraint inequality: 0 <= lambda_i <= C
        if self.C:
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        else:
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))

        # Constraint equality: sum(lambda_i * y_i) = 0
        A = cvxopt.matrix(y_binary.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = self.max_iter

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        lambdas = np.ravel(sol['x'])

        # Find support vectors (those with non-zero Lagrange multipliers)
        if self.C:
            is_sv = (lambdas >= 1e-5) & (lambdas <= self.C)
        else:
            is_sv = lambdas >= 1e-5

        self.support_vectors_ = X[is_sv]
        self.support_labels_ = y_binary[is_sv]
        self.dual_coef_ = lambdas[is_sv]

        # Compute bias as b = 1/N_s sum_i{y_i - sum_sv{lambda_sv * y_sv * K(x_sv, x_i)}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self.intercept_ = 0.0
        for i in range(len(self.dual_coef_)):
            self.intercept_ += self.support_labels_[i]
            self.intercept_ -= np.sum(self.dual_coef_ * self.support_labels_ * K[sv_index[i], is_sv])
        self.intercept_ /= len(self.dual_coef_)

        # Compute weights for linear kernel
        if self.kernel == 'linear':
            self.coef_ = np.zeros(n_features)
            for i in range(len(self.dual_coef_)):
                self.coef_ += self.dual_coef_[i] * self.support_vectors_[i] * self.support_labels_[i]
        else:
            self.coef_ = None

    def _fit_smo(self, X, y_binary, n_samples, n_features):
        kernel_fn = self._get_kernel_fn()

        # Initialize alphas and bias (use array for mutable reference)
        alphas = np.zeros(n_samples)
        bias = np.array([0.0])

        # Compute initial errors
        errors = self._compute_errors(alphas, bias[0], X, y_binary, kernel_fn)

        # SMO main loop
        num_changed = 0
        examine_all = True

        for _ in range(self.max_iter):
            num_changed = 0

            if examine_all:
                for i in range(n_samples):
                    num_changed += self._examine_example(i, X, y_binary, alphas, bias, errors, kernel_fn, n_samples)
            else:
                for i in np.where((alphas != 0) & (alphas != self.C))[0]:
                    num_changed += self._examine_example(i, X, y_binary, alphas, bias, errors, kernel_fn, n_samples)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            if examine_all and num_changed == 0:
                break

        # Extract support vectors
        is_sv = alphas > 1e-5
        self.support_vectors_ = X[is_sv]
        self.support_labels_ = y_binary[is_sv]
        self.dual_coef_ = alphas[is_sv]
        self.intercept_ = bias[0]

        # Compute weights for linear kernel
        if self.kernel == 'linear':
            self.coef_ = np.zeros(n_features)
            for i in range(len(self.dual_coef_)):
                self.coef_ += self.dual_coef_[i] * self.support_vectors_[i] * self.support_labels_[i]
        else:
            self.coef_ = None

    def _examine_example(self, i2, X, y, alphas, bias, errors, kernel_fn, n_samples):
        y2 = y[i2]
        alph2 = alphas[i2]
        E2 = errors[i2]
        r2 = E2 * y2

        # Check KKT conditions
        if (r2 < -self.tol and alph2 < self.C) or (r2 > self.tol and alph2 > 0):
            non_bound_indices = np.where((alphas != 0) & (alphas != self.C))[0]

            # Heuristic: choose max error difference
            if len(non_bound_indices) > 1:
                if E2 > 0:
                    i1 = np.argmin(errors)
                else:
                    i1 = np.argmax(errors)

                if self._take_step(i1, i2, X, y, alphas, bias, errors, kernel_fn, n_samples):
                    return 1

            # Loop through non-bound alphas
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state)
                start_idx = rng.choice(n_samples)
            else:
                start_idx = np.random.choice(n_samples)

            for i1 in np.roll(non_bound_indices, start_idx):
                if self._take_step(i1, i2, X, y, alphas, bias, errors, kernel_fn, n_samples):
                    return 1

            # Loop through all alphas
            for i1 in np.roll(np.arange(n_samples), start_idx):
                if self._take_step(i1, i2, X, y, alphas, bias, errors, kernel_fn, n_samples):
                    return 1

        return 0

    def _take_step(self, i1, i2, X, y, alphas, bias, errors, kernel_fn, n_samples):
        if i1 == i2:
            return False

        b = bias[0]
        alph1 = alphas[i1]
        alph2 = alphas[i2]
        y1 = y[i1]
        y2 = y[i2]
        E1 = errors[i1]
        E2 = errors[i2]
        s = y1 * y2

        # Compute L and H bounds
        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)

        if L == H:
            return False

        # Compute eta (second derivative)
        k11 = kernel_fn(X[i1], X[i1])
        k12 = kernel_fn(X[i1], X[i2])
        k22 = kernel_fn(X[i2], X[i2])
        eta = k11 + k22 - 2 * k12

        # Compute new alpha2
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # Use objective function at boundaries
            alphas_temp = alphas.copy()
            alphas_temp[i2] = L
            Lobj = self._objective_function(alphas_temp, y, kernel_fn, X)
            alphas_temp[i2] = H
            Hobj = self._objective_function(alphas_temp, y, kernel_fn, X)

            eps = 1e-3
            if Lobj < Hobj - eps:
                a2 = L
            elif Lobj > Hobj + eps:
                a2 = H
            else:
                a2 = alph2

        # Check for sufficient change
        if np.abs(a2 - alph2) < 1e-5 * (a2 + alph2 + 1e-5):
            return False

        # Compute new alpha1
        a1 = alph1 + s * (alph2 - a2)

        # Update bias
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b

        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) * 0.5

        # Update alphas
        alphas[i1] = a1
        alphas[i2] = a2

        # Update errors
        for i in [i1, i2]:
            if 0 < alphas[i] < self.C:
                errors[i] = 0.0

        non_opt = [n for n in range(n_samples) if n != i1 and n != i2]
        if len(non_opt) > 0:
            errors[non_opt] += y1 * (a1 - alph1) * kernel_fn(X[i1], X[non_opt]) + \
                                y2 * (a2 - alph2) * kernel_fn(X[i2], X[non_opt]) + b - b_new

        # Update bias array
        bias[0] = b_new

        return True

    def _compute_errors(self, alphas, b, X, y, kernel_fn):
        n_samples = len(y)
        errors = np.zeros(n_samples)
        for i in range(n_samples):
            prediction = np.sum(alphas * y * kernel_fn(X, X[i])) + b
            errors[i] = prediction - y[i]
        return errors

    def _objective_function(self, alphas, y, kernel_fn, X):
        K = self._compute_kernel_matrix(X, kernel_fn)
        return -np.sum(alphas) + 0.5 * np.sum(
            (y[:, np.newaxis] * y[np.newaxis, :]) * K * (alphas[:, np.newaxis] * alphas[np.newaxis, :])
        )

    def _fit_pegasos(self, X, y_binary, n_samples, n_features):
        kernel_fn = self._get_kernel_fn()

        # Initialize alphas
        alphas = np.zeros(n_samples)

        # Set random state
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()

        # Pegasos iterations
        for t in range(1, self.max_iter + 1):
            i = rng.randint(0, n_samples)

            # Compute decision value
            decision_val = 0.0
            for j in range(n_samples):
                decision_val += alphas[j] * y_binary[i] * kernel_fn(X[i], X[j])

            decision_val *= y_binary[i] / (self.C * t)

            # Update alpha if margin violated
            if decision_val < 1:
                alphas[i] += 1

        # Extract support vectors
        is_sv = alphas > 0
        self.support_vectors_ = X[is_sv]
        self.support_labels_ = y_binary[is_sv]
        self.dual_coef_ = alphas[is_sv]
        self.intercept_ = 0.0

        # Compute weights for linear kernel
        if self.kernel == 'linear':
            self.coef_ = np.zeros(n_features)
            for i in range(len(self.dual_coef_)):
                self.coef_ += self.dual_coef_[i] * self.support_vectors_[i] * self.support_labels_[i]
        else:
            self.coef_ = None

    def _decision_function(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        kernel_fn = self._get_kernel_fn()

        if self.coef_ is not None:
            return np.dot(X, self.coef_) + self.intercept_
        else:
            y_predict = np.zeros(len(X))
            for k in range(len(X)):
                for lda, sv_x, sv_y in zip(self.dual_coef_, self.support_vectors_, self.support_labels_):
                    y_predict[k] += lda * sv_y * kernel_fn(X[k], sv_x)
            return y_predict + self.intercept_

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        decision = self._decision_function(X)
        y_pred = np.where(np.sign(decision) == -1, self.classes_[0], self.classes_[1])
        return y_pred

    def _check_is_fitted(self):
        if not self.isfitted:
            raise ValueError("You must call `fit` before `predict`")
