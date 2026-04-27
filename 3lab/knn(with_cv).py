import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('data/data180.csv', sep=';', encoding='utf-8')

df = df.drop(columns=['продукт'])

X = df.iloc[:, 0:5]

y = df.iloc[:, 5]

scaler = StandardScaler()
knn = KNeighborsClassifier()

knn_pipeline = make_pipeline(scaler, knn)

param_grid = {
    'kneighborsclassifier__n_neighbors': [i for i in range(1,11)],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(knn_pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)
print(f"Best params for KNN: {grid_search.best_params_}")
print(f"Best score using GridSearchCV: {grid_search.best_score_}")

best_model = grid_search.best_estimator_

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
k_fold_cv_score = cross_val_score(best_model, X, y, cv=k_fold, scoring='accuracy')
mean_accuracy = np.average(k_fold_cv_score)
print("================================")
print(f"K-Fold CV Scores: {k_fold_cv_score}")
print(f"Средняя точность: {mean_accuracy:.4f}")

t0 = timeit.default_timer()
predict = cross_val_predict(best_model, X, y, cv=k_fold)
pred_time_s = timeit.default_timer() - t0
print(f"Время предсказания: {pred_time_s * 1000:.3f} ms")


class_report = classification_report(y, predict)
print("================================")
print(f"Отчет классификации:\n  {class_report}")

conf_matrix = confusion_matrix(y, predict)
print(f"Матрица ошибок:\n {conf_matrix}")

result = permutation_importance(grid_search.best_estimator_, X, y, n_repeats=10, random_state=42)
importance = result.importances_mean

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Предсказанный класс')
plt.ylabel('Настоящий класс')
plt.title('Матрица ошибок')
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(X.columns, importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance (Permutation)')
plt.show()