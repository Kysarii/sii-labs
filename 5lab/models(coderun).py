import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression, ARDRegression, RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

train_df = pd.read_csv('data/train.csv', sep='\t', header=None)
test_df = pd.read_csv('data/test.csv', sep='\t', header=None)

print('Размер train:', train_df.shape)
print('Размер test:', test_df.shape)
print('\nПервые строки train:')
print(train_df.head())

X = train_df.iloc[:, :100]
y = train_df.iloc[:, 100]
X_test = test_df.iloc[:, :100]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#param_grid = {
#    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#    "fit_intercept": [True, False],
#    "solver": ["auto", "svd", "cholesky", "lsqr"]
#}
#
#grid = GridSearchCV(
#    estimator=Ridge(),
#    param_grid=param_grid,
#    scoring="neg_mean_squared_error",
#    cv=5,
#    n_jobs=-1
#)
#
#grid.fit(X_train, y_train)
#best_model = grid.best_estimator_
#valid_pred = best_model.predict(X_valid)

ridge = Ridge()
ridge.fit(X_train, y_train)
valid_pred = ridge.predict(X_valid)
mae = mean_absolute_error(y_valid, valid_pred)
mse = mean_squared_error(y_valid, valid_pred)
r2 = r2_score(y_valid, valid_pred)

print(f'MAE: {mae:.6f}')
print(f'MSE: {mse:.6f}')
print(f'R2: {r2:.6f}')

test_pred = ridge.predict(X_test)

print('\nПервые предсказания для test:')
print(np.round(test_pred[:10], 6))

pd.DataFrame(test_pred).to_csv("answer.tsv", sep="\t", index=False, header=False, float_format="%.8f")
print("Готово: answer.csv сохранен")

plt.figure(figsize=(6, 5))
plt.scatter(y_valid, valid_pred, alpha=0.5, color='steelblue', label='Предсказания')
mn = min(y_valid.min(), valid_pred.min())
mx = max(y_valid.max(), valid_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--', label='Идеал: y = x')
plt.xlabel('Настоящая цена')
plt.ylabel('Предсказанная цена')
plt.title('Факт vs предсказание')
plt.legend()
plt.tight_layout()
plt.show()

residuals = y_valid - valid_pred
plt.figure(figsize=(6, 5))
sns.histplot(residuals, bins=30, kde=True, color='blue')
plt.xlabel('Ошибка')
plt.ylabel('Количество')
plt.title('Распределение ошибок')
plt.tight_layout()
plt.show()