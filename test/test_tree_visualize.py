import pandas as pd
from src.allyouneed.tree.decision_tree_classifier import DecisionTreeClassifier

data = pd.read_csv("dataset/train.csv")

df = pd.DataFrame(data)
X = df.drop(columns=['Target']).iloc[:,0:3] # dikit dlu biar ga jelek gambarnya
y = df['Target'].values


print("Melatih Decision Tree Classifier...")

clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion='gini')

feature_names = list(X.columns)
clf.fit(X, y, feature_names=feature_names)

print("Training selesai.")

output_path = input("Masukkan nama file output: ")
top_n = int(input("Masukkan max depth yang ingin ditampilkan: "))

clf.visualize_tree(f"test/output/{output_path}.png", top_n=top_n)

print(f"File saved to test/output/{output_path}.png!")
