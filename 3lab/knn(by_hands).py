import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [
                np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train     #евклидово расстояние
                #np.sum(np.abs(x - x_train)) for x_train in self.X_train            #расстояние манхэттена
            ]
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

df = pd.read_csv("data/data180.csv", sep=";", encoding="utf-8")

df = df.drop(columns=["продукт"])

X = df.iloc[:, 0:5]
y = df.iloc[:, 5]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=11, test_size=0.2
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = KNN(k=5)
clf.fit(X_train, y_train)

t0 = timeit.default_timer()
predictions = clf.predict(X_test)
pred_time_s = timeit.default_timer() - t0
print(f"Время предсказания: {pred_time_s * 1000:.3f} ms")

acc = np.mean(np.asarray(predictions) == np.asarray(y_test))
print(f"Средняя точность:  {acc}")

cr = classification_report(y_test, predictions)
print("================================")
print(f"Отчет классификации:\n  {cr}")

cm = confusion_matrix(y_test, predictions)
print(f"Матрица Ошибок:\n {cm}")

print("Реальные метки (y_test):")
print(y_test.values)
print("Предсказанные метки (y_pred):")
print(predictions)

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