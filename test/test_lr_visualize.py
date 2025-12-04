import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from src.allyouneed.linear_model.logistic import LogisticRegression

def generate_data(n_samples=300):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_clusters_per_class=1, 
        random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def compute_metric(w, X, y, mode="loss"):
    z = np.dot(X, w)
    z = np.clip(z, -500, 500)
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    ll = np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    if mode == "loss":
        return -ll
    return ll

def visualize_training(model, X, y, save_path=None):
    if not hasattr(model, 'history') or not model.history:
        return

    solver_name = getattr(model, 'solver_name', 'mgd')
    
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
        }
    }
    
    config = configs.get(solver_name, configs["mgd"])
    mode = config["mode"]
    
    history = np.array(model.history)
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
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
            Z[i, j] = compute_metric(w_tmp, X_bias, y, mode=mode)

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
    plt.show()
    
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=30)

if __name__ == "__main__":
    X, y = generate_data()
    
    print("Testing 1: Mini-Batch Gradient Descent (Loss Minimization)")
    model_mgd = LogisticRegression(learning_rate=0.1, max_iter=100, solver="mgd", batch_size=16, tol=1e-4)
    model_mgd.fit(X, y)
    visualize_training(model_mgd, X, y)

    print("Testing 2: Stochastic Gradient Ascent (Likelihood Maximization)")
    model_sga = LogisticRegression(learning_rate=0.1, max_iter=100, solver="sga", batch_size=16, tol=1e-4)
    model_sga.fit(X, y)
    visualize_training(model_sga, X, y)