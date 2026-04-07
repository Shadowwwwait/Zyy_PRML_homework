import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
def make_moons_3d(n_samples=500, noise=0.2):
    t = np.linspace(0, 2*np.pi, n_samples)
    x = 1.5*np.cos(t)
    y = np.sin(t)
    z = np.sin(2*t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y-1, -z])])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集类别分布: {np.bincount(y_train.astype(int))}")
print(f"测试集类别分布: {np.bincount(y_test.astype(int))}")
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost+DT': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42),
    'SVM (linear)': SVC(kernel='linear', random_state=42),
    'SVM (rbf)': SVC(kernel='rbf', gamma='scale', random_state=42),
    'SVM (poly)': SVC(kernel='poly', degree=3, gamma='scale', random_state=42)
}
results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name}")
    print(f"准确率: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['C0', 'C1']))
plt.figure(figsize=(10,6))
names = list(results.keys())
accs = list(results.values())
plt.bar(names, accs, color=['skyblue', 'lightgreen', 'orange', 'red', 'purple'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Classifier Performance on 3D Moons Dataset')
for i, v in enumerate(accs):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center')
plt.xticks(rotation=15)
plt.show()































