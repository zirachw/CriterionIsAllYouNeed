import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from allyouneed.linear_model.logistic import LogisticRegression

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

if __name__ == "__main__":
    X, y = generate_data()
    
    print("Testing 1: Mini-Batch Gradient Descent (Loss Minimization)")
    model_mgd = LogisticRegression(learning_rate=0.1, max_iter=100, solver="mgd", batch_size=16, tol=1e-4)
    model_mgd.fit(X, y)
    model_mgd.visualize_training(save_path="test/output/mgd_training.mp4")

    print("Testing 2: Stochastic Gradient Ascent (Likelihood Maximization)")
    model_sga = LogisticRegression(learning_rate=0.1, max_iter=100, solver="sga", batch_size=16, tol=1e-4)
    model_sga.fit(X, y)
    model_sga.visualize_training(save_path="test/output/sga_training.mp4")
    
    print("Testing 3: Batch Gradient Ascent (Likelihood Maximization)")
    model_bga = LogisticRegression(learning_rate=0.1, max_iter=100, solver="bga", tol=1e-4)
    model_bga.fit(X, y)
    model_bga.visualize_training(save_path="test/output/bga_training.mp4")

    print("Testing 4: Newton's Method (Likelihood Maximization)")
    model_newton = LogisticRegression(learning_rate=1.0, max_iter=20, solver="newton-cg", tol=1e-4)
    model_newton.fit(X, y)
    model_newton.visualize_training(save_path="test/output/newton_training.mp4")