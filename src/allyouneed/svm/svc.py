import numpy as np
import cvxopt
from ..base import BaseClassifier


class SVC(BaseClassifier):
    def __init__(self, kernel='linear', gamma=None, degree=3, coef0=0.0, C=1.0, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
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
        cvxopt.solvers.options['maxiters'] = 200

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

        self.isfitted = True
        return self

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
