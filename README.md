# CriterionIsAllYouNeed

Implementasi mini-library machine learning (terinpirasi oleh scikit-learn) untuk kebutuhan klasifikasi dan seleksi fitur. Di dalamnya ada Logistic Regression, Support Vector Classifier dengan berbagai kernel, Decision Tree Classifier (Gini/Entropy + visualisasi), PCA, seleksi fitur forward/backward/backward-forward, K-Fold/StratifiedKFold, encoder & scaler sederhana, serta metrik Accuracy dan F1. Proyek ini dipakai untuk eksperimen klasifikasi data mahasiswa (`dataset/train.csv` & `dataset/test.csv`) pada notebook `src/main.ipynb`.

## Struktur Singkat
- `dataset/`: data latih/uji dan `sample_submission.csv`.
- `src/allyouneed/`: modul utama (preprocessing, feature_selection, linear_model, svm, tree, metrics, model_selection, decomposition).
- `src/main.ipynb`: notebook eksplorasi/pelatihan.
- `test/`: skrip visualisasi Logistic Regression dan Decision Tree (`test/output/` menyimpan hasil gambar).

## Prasyarat & Instalasi
Pastikan Python 3.10+ tersedia.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn cvxopt notebook
```
Catatan: `ffmpeg` diperlukan jika ingin menyimpan animasi training Logistic Regression ke video dari matplotlib.

## Cara Menjalankan
- Notebook: `jupyter notebook src/main.ipynb` lalu jalankan sel sesuai kebutuhan.
- Visualisasi lintasan bobot Logistic Regression: `python test/test_lr_visualize.py` (menggunakan data sintetis scikit-learn).
- Visualisasi pohon keputusan pada data `dataset/train.csv`: `python test/test_tree_visualize.py` lalu isi prompt nama file output dan kedalaman visualisasi; hasil tersimpan di `test/output/<nama>.png`.
- Menggunakan library di kode Python:
  ```python
  import numpy as np
  from src.allyouneed.linear_model import LogisticRegression
  from src.allyouneed.metrics import Accuracy

  X = np.array([[0.2, 1.1], [1.0, 0.3], [0.8, 0.9], [1.2, 1.4]])
  y = np.array([0, 0, 1, 1])

  model = LogisticRegression(max_iter=200, solver="mgd", learning_rate=0.1)
  model.fit(X, y)
  pred = model.predict(X)
  acc = Accuracy()(y, pred)
  print(f"Train accuracy: {acc:.3f}")
  ```

## Pembagian Tugas Anggota
| Nama | NIM | Tugas |
| --- | --- | --- |
| Adhimas Aryo Bimo | 13523052 | Logistic Regression, Laporan |
| Muhammad Fathur Rizky | 13523105 | Metric, DecisionTreeClassifier (CART), Preprocessing, Laporan |
| Guntara Hambali | 13523114 | Bonus Video Logistic, Plot DecisionTreeClassifier, Laporan |
| Razi Rachman Widyadhana | 13523004 | SVC (Linear, RBF), Laporan |
| Ahmad Wicaksono | 13523121 | Forward, Backward, Laporan |
