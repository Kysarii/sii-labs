import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report

dataset_train = pd.read_csv("data/disease_train.csv", sep=",", encoding="utf-8")
dataset_test = pd.read_csv("data/disease_public_test.csv", sep=",", encoding="utf-8")
dataset_submission = pd.read_csv("data/disease_sample_submission.csv", sep=",", encoding="utf-8")

X_train = dataset_train[["X1", "X2", "X3", "X4", "X5", "X6", "X7"]]
y_train = dataset_train["Y"]
print("Классы в обучающей выборке: ", y_train.value_counts())

X_test = dataset_test[["X1", "X2", "X3", "X4", "X5", "X6", "X7"]]
y_test_true = dataset_submission["Y"]
print("Классы в тестовой выборке: ", y_test_true.value_counts())

lr_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(
        cv=10,
        random_state=42,
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        l1_ratios=(0.0,),
        use_legacy_attributes=True,
    ),
)

lr_pipeline.fit(X_train, y_train)
lrcv = lr_pipeline[-1]

print("Выбранное C:", lrcv.C_)

sc = lrcv.scores_
scores_mat = np.asarray(next(iter(sc.values())))

best_c_idx = int(np.argmax(scores_mat.mean(axis=0)))
acc_per_fold = np.ravel(scores_mat[:, best_c_idx])
print("================================")
print("Внутренняя CV (LogisticRegressionCV), accuracy по фолдам (train):")
print(acc_per_fold)
print(f"Средняя accuracy: {acc_per_fold.mean():.4f}, std: {acc_per_fold.std():.4f}")

fold_labels = [f"Fold {i + 1}" for i in range(len(acc_per_fold))]
x = np.arange(len(fold_labels))
plt.figure(figsize=(8, 5))
plt.bar(x, acc_per_fold, color="steelblue")
plt.xticks(x, fold_labels)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Внутренняя CV: accuracy по фолдам (train)")
plt.tight_layout()
plt.show()

y_pred_train = lr_pipeline.predict(X_train)
print("================================")
print("Отчёт классификации (train)")
print(classification_report(y_train, y_pred_train))

conf_matrix_train = confusion_matrix(y_train, y_pred_train)
print(f"Матрица ошибок (train):\n{conf_matrix_train}")

result = permutation_importance(lr_pipeline, X_train, y_train, n_repeats=10, random_state=42)
importance = result.importances_mean

y_pred_test = lr_pipeline.predict(X_test)

lr_pipeline.fit(X_test, y_test_true)
lrcv_test = lr_pipeline[-1]

sc_test = lrcv_test.scores_
scores_mat_test = np.asarray(next(iter(sc_test.values())))

best_c_idx_test = int(np.argmax(scores_mat_test.mean(axis=0)))
test_acc_per_fold = np.ravel(scores_mat_test[:, best_c_idx_test])
print("================================")
print("Внутренняя CV (LogisticRegressionCV), accuracy по фолдам (test):")
print(test_acc_per_fold)
print(
    f"Средняя accuracy: {test_acc_per_fold.mean():.4f}, std: {test_acc_per_fold.std():.4f}"
)

fold_labels = [f"Fold {i + 1}" for i in range(len(test_acc_per_fold))]
x = np.arange(len(fold_labels))
plt.figure(figsize=(8, 5))
plt.bar(x, test_acc_per_fold, color="steelblue")
plt.xticks(x, fold_labels)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Внутренняя CV: accuracy по фолдам (test)")
plt.tight_layout()
plt.show()

print("================================")
print("Отчет классификации: предсказания vs эталон (disease_sample_submission.csv, столбец Y)")
print(classification_report(y_test_true, y_pred_test))

conf_matrix_test = confusion_matrix(y_test_true, y_pred_test)
print(f"Матрица ошибок (test):\n{conf_matrix_test}")


plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix_train,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["0", "1"],
    yticklabels=["0", "1"],
)
plt.xlabel("Предсказанный класс")
plt.ylabel("Настоящий класс")
plt.title("Матрица ошибок (train)")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["0", "1"],
    yticklabels=["0", "1"],
)
plt.xlabel("Предсказанный класс")
plt.ylabel("Эталонный класс")
plt.title("Матрица ошибок (test vs эталон)")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(X_train.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()
