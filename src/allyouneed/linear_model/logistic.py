import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..base import BaseClassifier


class LogisticRegression(BaseClassifier):
    SOLVER_TABLE = {
        "sga": "_stochastic_gradient_ascent",
        "bga": "_batch_gradient_ascent",
        "mgd": "_mini_batch_gradient_descent",
        "newton-cg": "_solve_newton_cg",
    }
    
    def __init__(
        self,
        max_iter=200,
        learning_rate=0.1,
        solver="sga",
        batch_size=32,
        random_state=None,
        tol=1e-4,
        class_weight=None,
        C=1.0,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        if solver not in self.SOLVER_TABLE:
            raise ValueError(f"Unknown solver: {solver}")
        self.solver_name = solver
        self.solver_fn = getattr(self, self.SOLVER_TABLE[solver])
        self.batch_size = batch_size
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.tol = tol
        self.class_weight = class_weight
        self.C = C

        self._classes = None
        self._W = None
        self._isfitted = False
        self.history = []
        
    def _logistic_function(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_sample_weight(self, y):
        if self.class_weight == 'balanced':
            classes, class_counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            weights = n_samples / (n_classes * class_counts)
            sample_weight = weights[np.searchsorted(classes, y)]
        elif isinstance(self.class_weight, dict):
            sample_weight = np.array([self.class_weight.get(cls, 1.0) for cls in y])
        else:
            sample_weight = np.ones(len(y))
        return sample_weight

    def _check_is_fitted(self):
        if not self._isfitted:
            raise ValueError("You must call `fit` before `predict`.")
            
    def _validate_input(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            return X, y
        return X
        
    def _add_bias(self, X):
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])
        
    def _stochastic_gradient_ascent(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        self.history = [w.copy()]  # Initialize history

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            indices = self._rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                w_batch = sample_weight[batch_idx]

                p = self._logistic_function(X_batch @ w)
                error = (y_batch - p) * w_batch
                gradient = X_batch.T @ error / w_batch.sum()
                w += self.learning_rate * gradient
                self.history.append(w.copy())  # Track each update

            # cek semuanya
            p_full = self._logistic_function(X @ w)
            error_full = (y - p_full) * sample_weight
            gradient_full = X.T @ error_full / sample_weight.sum()
            if np.linalg.norm(gradient_full) < self.tol:
                break

        return w
    
    def _compute_loss(self, w, X, y, sample_weight, alpha):
        z = X @ w
        p = self._logistic_function(z)
        
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        data_loss = -np.sum(sample_weight * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        reg_loss = 0.5 * alpha * np.dot(w, w)
        loss = data_loss + reg_loss
        
        return loss

    def _compute_gradient(self, w, X, y, sample_weight, alpha):
        z = X @ w
        p = self._logistic_function(z)
        
        error = (p - y) * sample_weight
        grad = X.T @ error + alpha * w
        
        return grad

    def _compute_hessian(self, w, X, y, sample_weight, alpha):
        z = X @ w
        p = self._logistic_function(z)
        
        S = p * (1 - p) * sample_weight
        H = X.T @ (X * S[:, np.newaxis])
        np.fill_diagonal(H, H.diagonal() + alpha)
        
        return H

    def _solve_newton_cg(self, X, y, sample_weight=None):
        n_features = X.shape[1]
        w = np.zeros(n_features)
        self.history = [w.copy()]  # Initialize history
        
        alpha = 1.0 / self.C if self.C > 0 else 1e-4

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        for iteration in range(self.max_iter):
            grad = self._compute_gradient(w, X, y, sample_weight, alpha)
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            H = self._compute_hessian(w, X, y, sample_weight, alpha)
            
            delta = self._conjugate_gradient(H, -grad, tol=1e-5, max_iter=min(n_features, 20))
            
            step_size = self._line_search(w, delta, X, y, sample_weight, alpha)
            
            w = w + step_size * delta
            self.history.append(w.copy())  # Track each iteration

        return w
    
    def _conjugate_gradient(self, A, b, tol=1e-5, max_iter=None):
        n = len(b)
        if max_iter is None:
            max_iter = n
        
        x = np.zeros(n)
        r = b.copy()
        p = r.copy()
        rs_old = np.dot(r, r)
        
        for _ in range(max_iter):
            if np.sqrt(rs_old) < tol:
                break
            
            Ap = A @ p
            alpha = rs_old / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = np.dot(r, r)
            
            if np.sqrt(rs_new) < tol:
                break
            
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        
        return x
    
    def _line_search(self, w, direction, X, y, sample_weight, alpha, max_steps=10):
        step_size = 1.0
        c = 0.5
        rho = 0.8
        
        current_loss = self._compute_loss(w, X, y, sample_weight, alpha)
        grad = self._compute_gradient(w, X, y, sample_weight, alpha)
        expected_decrease = c * np.dot(grad, direction)
        
        for _ in range(max_steps):
            new_w = w + step_size * direction
            new_loss = self._compute_loss(new_w, X, y, sample_weight, alpha)
            
            if new_loss <= current_loss + step_size * expected_decrease:
                break
            
            step_size *= rho
        
        return step_size
    
    def _batch_gradient_ascent(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        self.history = [w.copy()]  # Initialize history

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            p = self._logistic_function(X @ w)
            error = (y - p) * sample_weight
            gradient = X.T @ error / sample_weight.sum()

            w += self.learning_rate * gradient
            self.history.append(w.copy())  # Track each iteration

            if np.linalg.norm(gradient) < self.tol:
                break

        return w
    
    def _mini_batch_gradient_descent(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        self.history = [w.copy()]  # Initialize history

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            indices = self._rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                wb = sample_weight[batch_idx]

                p = self._logistic_function(Xb @ w)
                error = (p - yb) * wb
                gradient = Xb.T @ error / wb.sum()

                w -= self.learning_rate * gradient  # descent
                self.history.append(w.copy())  # Track each update

            # cek konvergensi full batch
            p_full = self._logistic_function(X @ w)
            error_full = (p_full - y) * sample_weight
            gradient_full = X.T @ error_full / sample_weight.sum()
            if np.linalg.norm(gradient_full) < self.tol:
                break

        return w

        
    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        self._classes = np.unique(y)
        X_bias = self._add_bias(X)

        if self.class_weight is not None:
            sample_weight = self._compute_sample_weight(y)
        else:
            sample_weight = np.ones(len(y))

        if self._classes.size > 2:
            W_all = []
            history_all = []
            for cls in self._classes:
                y_binary = (y == cls).astype(int)
                w = self.solver_fn(X_bias, y_binary, sample_weight)
                W_all.append(w)
                history_all.append(self.history.copy() if self.history else [])
            self._W = np.vstack(W_all)
            self._training_data = (X_bias, y)
            self._class_histories = history_all

        else:
            positive_class = self._classes[-1]
            y_binary = (y == positive_class).astype(int)
            self._W = self.solver_fn(X_bias, y_binary, sample_weight)
            self._training_data = (X_bias, y_binary)

        self._isfitted = True
        return self
        
    def predict_proba(self, X):
        self._check_is_fitted()
        X = self._add_bias(self._validate_input(X))
        
        if self._classes.size > 2:
            logits = X @ self._W.T
            return self._logistic_function(logits)
        
        logits = X @ self._W
        proba_pos = self._logistic_function(logits)
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])
    
    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred = np.argmax(probs, axis=1)
        return self._classes[y_pred]
    
    def _compute_metric(self, w, X, y, mode="loss"):
        z = np.dot(X, w)
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        ll = np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        if mode == "loss":
            return -ll
        return ll
    
    def visualize_training(self, save_path=None, class_idx=None):
        if not self._isfitted:
            raise ValueError("Model must be fitted before visualization.")
        
        X_train, y_train = self._training_data
        
        if self._classes.size > 2:
            if not hasattr(self, '_class_histories'):
                raise ValueError("No training history available for multiclass model.")
            return self._visualize_multiclass(X_train, y_train, save_path, class_idx)
        else:
            if not self.history:
                raise ValueError("No training history available. Train the model first.")
            return self._visualize_binary(X_train, y_train, save_path)
    
    def _visualize_binary(self, X_train, y_train, save_path=None):
        configs = {
            "mgd": {
                "mode": "loss",
                "title": "Loss Landscape (Gradient Descent)",
                "cmap": "viridis" 
            },
            "sga": {
                "mode": "likelihood",
                "title": "Likelihood Landscape (Gradient Ascent)",
                "cmap": "viridis"
            },
            "bga": {
                "mode": "likelihood",
                "title": "Likelihood Landscape (Gradient Ascent)",
                "cmap": "viridis"
            },
            "newton-cg": {
                "mode": "loss",
                "title": "Loss Landscape (Newton-CG)",
                "cmap": "viridis"
            }
        }
        
        config = configs.get(self.solver_name, configs["mgd"])
        mode = config["mode"]
        
        history = np.array(self.history)
        final_w = history[-1]
        w0_h, w1_h = history[:, 0], history[:, 1]

        span_w0 = np.ptp(w0_h)
        span_w1 = np.ptp(w1_h)
        pad = 0.5 if span_w0 > 0.1 else 1.0

        w0_vals = np.linspace(np.min(w0_h)-span_w0*pad, np.max(w0_h)+span_w0*pad, 50)
        w1_vals = np.linspace(np.min(w1_h)-span_w1*pad, np.max(w1_h)+span_w1*pad, 50)
        W0, W1 = np.meshgrid(w0_vals, w1_vals)

        Z = np.zeros_like(W0)
        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                w_tmp = final_w.copy()
                w_tmp[0], w_tmp[1] = W0[i, j], W1[i, j]
                Z[i, j] = self._compute_metric(w_tmp, X_train, y_train, mode=mode)

        fig, ax = plt.subplots(figsize=(9, 6))
        CS = ax.contourf(W0, W1, Z, 50, cmap=config['cmap'])
        cbar = plt.colorbar(CS)
        cbar.set_label(f"{mode.capitalize()} Value")

        line, = ax.plot([], [], 'r-', lw=2, label='Path')
        point, = ax.plot([], [], 'ro', markersize=8, markeredgecolor='white')

        ax.plot(w0_h[0], w1_h[0], 'bx', markersize=10, markeredgewidth=2, label='Start')
        ax.plot(w0_h[-1], w1_h[-1], 'w*', markersize=12, markeredgecolor='black', label='End')

        ax.set_title(config['title'])
        ax.set_xlabel(r'$\theta_0$ (Bias)')
        ax.set_ylabel(r'$\theta_1$ (Feature 1)')
        ax.legend(loc='lower right')

        def update(frame):
            line.set_data(w0_h[:frame+1], w1_h[:frame+1])
            point.set_data([w0_h[frame]], [w1_h[frame]])
            return line, point

        ani = FuncAnimation(fig, update, frames=len(history), blit=True, interval=50)
        
        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=30)
        
        plt.show()
        return ani
    
    def _visualize_multiclass(self, X_train, y_train, save_path=None, class_idx=None):
        configs = {
            "mgd": {"mode": "loss", "title": "Loss Landscape (Gradient Descent)", "cmap": "viridis"},
            "sga": {"mode": "likelihood", "title": "Likelihood Landscape (Gradient Ascent)", "cmap": "viridis"},
            "bga": {"mode": "likelihood", "title": "Likelihood Landscape (Gradient Ascent)", "cmap": "viridis"},
            "newton-cg": {"mode": "loss", "title": "Loss Landscape (Newton-CG)", "cmap": "viridis"}
        }
        config = configs.get(self.solver_name, configs["mgd"])
        mode = config["mode"]
        
        if class_idx is not None:
            classes_to_plot = [class_idx]
        else:
            classes_to_plot = range(len(self._classes))
        
        n_classes = len(classes_to_plot)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows))
        if n_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_classes > 1 else [axes]
        
        animations = []
        
        for idx, cls_idx in enumerate(classes_to_plot):
            ax = axes[idx]
            history = np.array(self._class_histories[cls_idx])
            
            if len(history) == 0:
                ax.text(0.5, 0.5, f'No history for Class {self._classes[cls_idx]}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            y_binary = (y_train == self._classes[cls_idx]).astype(int)
            final_w = history[-1]
            w0_h, w1_h = history[:, 0], history[:, 1]
            
            span_w0 = np.ptp(w0_h)
            span_w1 = np.ptp(w1_h)
            pad = 0.5 if span_w0 > 0.1 else 1.0
            
            w0_vals = np.linspace(np.min(w0_h)-span_w0*pad, np.max(w0_h)+span_w0*pad, 50)
            w1_vals = np.linspace(np.min(w1_h)-span_w1*pad, np.max(w1_h)+span_w1*pad, 50)
            W0, W1 = np.meshgrid(w0_vals, w1_vals)
            
            Z = np.zeros_like(W0)
            for i in range(W0.shape[0]):
                for j in range(W0.shape[1]):
                    w_tmp = final_w.copy()
                    w_tmp[0], w_tmp[1] = W0[i, j], W1[i, j]
                    Z[i, j] = self._compute_metric(w_tmp, X_train, y_binary, mode=mode)
            
            CS = ax.contourf(W0, W1, Z, 50, cmap=config['cmap'])
            cbar = plt.colorbar(CS, ax=ax)
            cbar.set_label(f"{mode.capitalize()} Value")
            
            line, = ax.plot([], [], 'r-', lw=2, label='Path')
            point, = ax.plot([], [], 'ro', markersize=8, markeredgecolor='white')
            
            ax.plot(w0_h[0], w1_h[0], 'bx', markersize=10, markeredgewidth=2, label='Start')
            ax.plot(w0_h[-1], w1_h[-1], 'w*', markersize=12, markeredgecolor='black', label='End')
            
            ax.set_title(f"Class {self._classes[cls_idx]}: {config['title']}")
            ax.set_xlabel(r'$\theta_0$ (Bias)')
            ax.set_ylabel(r'$\theta_1$ (Feature 1)')
            ax.legend(loc='lower right')
            
            animations.append((line, point, w0_h, w1_h))
        
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        def update(frame):
            artists = []
            for line, point, w0_h, w1_h in animations:
                if frame < len(w0_h):
                    line.set_data(w0_h[:frame+1], w1_h[:frame+1])
                    point.set_data([w0_h[frame]], [w1_h[frame]])
                else:
                    line.set_data(w0_h, w1_h)
                    point.set_data([w0_h[-1]], [w1_h[-1]])
                artists.extend([line, point])
            return artists
        
        max_frames = max(len(hist) for hist in [self._class_histories[i] for i in classes_to_plot] if len(hist) > 0)
        ani = FuncAnimation(fig, update, frames=max_frames, blit=True, interval=50)
        
        plt.tight_layout()
        
        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=30)
        
        plt.show()
        return ani