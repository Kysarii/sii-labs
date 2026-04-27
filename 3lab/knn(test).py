import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report


class KNN:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        predictions = []
        for x in X:
            if self.metric == "euclidean":
                distances = [
                    np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train
                ]
            elif self.metric == "manhattan":
                distances = [
                    np.sum(np.abs(x - x_train)) for x_train in self.X_train
                ]
            else:
                raise ValueError(f"Неизвестная метрика: {self.metric}")
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions


def cv_fold_scores_and_oof(X, y, k_neighbors, metric, kf):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    scores = []
    oof_pred = np.empty(len(X), dtype=object)
    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        clf = KNN(k=k_neighbors, metric=metric)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_val_s)
        oof_pred[val_idx] = preds
        acc = np.mean(np.asarray(preds) == np.asarray(y_val))
        scores.append(acc)
    return scores, oof_pred


df = pd.read_csv("data/data180.csv", sep=";", encoding="utf-8")
df = df.drop(columns=["продукт"])
X = df.iloc[:, 0:5]
y = df.iloc[:, 5]

k_list = [i for i in range(1,11)]
metric_list = ["euclidean", "manhattan"]

kf = KFold(n_splits=10, shuffle=True, random_state=42)

best_k = None
best_metric = None
best_mean_cv = -1.0

print("=== Сетка по KFold (средняя точность по фолдам) ===")
for metric in metric_list:
    for k_neighbors in k_list:
        scores, _ = cv_fold_scores_and_oof(X, y, k_neighbors, metric, kf)
        mean_cv = float(np.mean(scores))
        print(f"  metric={metric}, k={k_neighbors:2d} -> фолды: {[round(s, 4) for s in scores]}, среднее: {mean_cv:.4f}")
        if mean_cv > best_mean_cv:
            best_mean_cv = mean_cv
            best_k = k_neighbors
            best_metric = metric

print()
print(f"Лучшие гиперпараметры: k={best_k}, metric={best_metric}, средняя CV={best_mean_cv:.4f}")

t0 = timeit.default_timer()
best_scores, oof_predictions = cv_fold_scores_and_oof(
    X, y, best_k, best_metric, kf
)
pred_time_s = timeit.default_timer() - t0

print()
print("=== Кросс-валидация для лучшей конфигурации ===")
for i, s in enumerate(best_scores, start=1):
    print(f"  Фолд {i}: {s:.4f}")
print(f"Средняя точность по кросс-валидации: {np.mean(best_scores):.4f}")
print(f"Время одного полного CV-прохода: {pred_time_s * 1000:.3f} ms")

cr = classification_report(y, oof_predictions)
print("================================")
print(f"Отчет классификации:\n{cr}")

cm = confusion_matrix(y, oof_predictions)
print(f"Матрица ошибок:\n{cm}")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sorted(y.unique()),
    yticklabels=sorted(y.unique()),
)
plt.xlabel("Предсказанный класс")
plt.ylabel("Настоящий класс")
plt.title("Матрица ошибок")
plt.show()
