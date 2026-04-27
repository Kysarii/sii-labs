import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('data/data180.csv', sep=';', encoding='utf-8')

df = df.drop(columns=['продукт'])

X = df.iloc[:, 0:5]
y = df.iloc[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn= KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

t0 = timeit.default_timer()
y_pred = knn.predict(X_test)
pred_time_s = timeit.default_timer() - t0
print(f"Время предсказания: {pred_time_s * 1000:.3f} ms")

acc = knn.score(X_test, y_test)
print(f"Средняя точность:  {acc}")

cr = classification_report(y_test, y_pred)
print("================================")
print(f"Отчет классификации:\n  {cr}")

cm = confusion_matrix(y_test, y_pred)
print(f"Матрица ошибок:\n {cm}")

print("Реальные метки (y_test):")
print(y_test.values)
print("Предсказанные метки (y_pred):")
print(y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Предсказанный класс')
plt.ylabel('Настоящий класс')
plt.title('Матрица ошибок')
plt.show()


